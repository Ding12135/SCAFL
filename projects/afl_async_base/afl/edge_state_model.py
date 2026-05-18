from __future__ import annotations

import math
from dataclasses import dataclass

from .runtime_state import ClientRuntimeState


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class EdgeObservation:
    compute_time_s: float
    upload_delay_s: float
    num_samples: int
    local_epochs: int
    update_payload_mb: float


class EdgeStateModel:
    """
    论文常见的端侧状态近似建模（server-side surrogate）：
    - CPU：按 local epochs / samples 的 CPU cycles 与观测 compute_time 估计
    - 网络：由真实模型更新 payload / upload_delay 估计 UL bandwidth，并递推 RTT/loss
    - 电池状态：CPU 计算能耗 + 通信能耗 + idle/base 能耗
    - 温度：一阶 RC 热模型
    - 内存：训练批量与 CPU 压力 proxy

    该模型不使用 AFL staleness 作为网络 RTT/loss 的直接输入，避免混淆
    “系统异步滞后”和“端侧通信状态”。
    """

    def __init__(self, cfg: dict):
        mc = cfg.get("edge_state_model", {}) if isinstance(cfg, dict) else {}

        self.enabled = bool(mc.get("enabled", True))
        self.dt_s = float(mc.get("dt_s", 1.0))
        self.ewma_alpha = float(mc.get("ewma_alpha", 0.85))

        # CPU proxy
        self.cpu_cycles_per_sample = float(mc.get("cpu_cycles_per_sample", 2.0e6))
        self.cpu_freq_min_ghz = float(mc.get("cpu_freq_min_ghz", 0.4))
        self.cpu_freq_max_ghz = float(mc.get("cpu_freq_max_ghz", 3.2))
        self.cpu_floor = float(mc.get("cpu_floor", 0.02))

        # Network proxy
        self.downlink_payload_ratio = float(mc.get("downlink_payload_ratio", 1.0))
        self.min_delay_s = float(mc.get("min_delay_s", 1e-3))
        self.bw_ul_min_mbps = float(mc.get("bw_ul_min_mbps", 0.2))
        self.bw_ul_max_mbps = float(mc.get("bw_ul_max_mbps", 300.0))
        self.bw_dl_min_mbps = float(mc.get("bw_dl_min_mbps", 0.2))
        self.bw_dl_max_mbps = float(mc.get("bw_dl_max_mbps", 600.0))
        self.rtt_floor_ms = float(mc.get("rtt_floor_ms", 8.0))
        self.rtt_upload_coeff_ms = float(mc.get("rtt_upload_coeff_ms", 120.0))

        # loss logistic
        self.loss_bias = float(mc.get("loss_bias", -3.0))
        self.loss_rtt_coeff = float(mc.get("loss_rtt_coeff", 1.1))
        self.loss_bw_coeff = float(mc.get("loss_bw_coeff", 0.8))
        self.loss_bw_ref_mbps = float(mc.get("loss_bw_ref_mbps", 10.0))
        self.loss_cap = float(mc.get("loss_cap", 0.6))

        # Energy / battery
        self.kappa_cpu = float(mc.get("kappa_cpu", 0.18))
        self.tx_power_w = float(mc.get("tx_power_w", 1.3))
        self.rx_power_w = float(mc.get("rx_power_w", 1.0))
        self.base_power_w = float(mc.get("base_power_w", 0.6))
        self.battery_mah = float(mc.get("battery_mah", 4500.0))
        self.battery_v = float(mc.get("battery_v", 3.85))

        # Thermal RC
        self.ambient_temp_c = float(mc.get("ambient_temp_c", 28.0))
        self.thermal_tau_s = float(mc.get("thermal_tau_s", 120.0))
        self.thermal_eta = float(mc.get("thermal_eta", 0.04))
        self.max_temp_c = float(mc.get("max_temp_c", 52.0))

        # Memory proxy
        self.mem_base = float(mc.get("mem_base", 0.25))
        self.mem_cpu_coeff = float(mc.get("mem_cpu_coeff", 0.55))
        self.mem_batch_coeff = float(mc.get("mem_batch_coeff", 0.20))
        self.mem_sample_ref = float(mc.get("mem_sample_ref", 1024.0))

    def _ewma(self, old: float, new: float) -> float:
        a = _clip(self.ewma_alpha, 0.0, 0.999)
        return a * float(old) + (1.0 - a) * float(new)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-float(x)))

    def _soc_drop_from_energy(self, e_joule: float) -> float:
        e_cap = (self.battery_mah / 1000.0) * self.battery_v * 3600.0
        if e_cap <= 0:
            return 0.0
        return float(e_joule) / float(e_cap)

    def update_client_state(self, cs: ClientRuntimeState, obs: EdgeObservation) -> None:
        if not self.enabled:
            return

        ct = max(0.0, float(obs.compute_time_s))
        ud = max(0.0, float(obs.upload_delay_s))
        ns = max(0, int(obs.num_samples))
        local_epochs = max(1, int(obs.local_epochs))
        ul_mb = max(0.0, float(obs.update_payload_mb))
        dl_mb = ul_mb * max(self.downlink_payload_ratio, 0.0)

        # 1) CPU utilization proxy from cycles/frequency, aligned with MEC compute models.
        total_cycles = self.cpu_cycles_per_sample * float(ns) * float(local_epochs)
        observed_freq_ghz = total_cycles / max(ct, self.min_delay_s) / 1.0e9
        cpu_inst = _clip(
            observed_freq_ghz / max(self.cpu_freq_max_ghz, 1e-9),
            self.cpu_floor,
            1.0,
        )
        cs.cpu_util_est = _clip(self._ewma(cs.cpu_util_est, cpu_inst), 0.0, 1.0)
        cpu_freq_ghz_est = _clip(observed_freq_ghz, self.cpu_freq_min_ghz, self.cpu_freq_max_ghz)

        # 2) Network state proxy from real model-update payload and upload delay.
        ul_bw_inst = 8.0 * ul_mb / max(ud, self.min_delay_s) if ul_mb > 0 else self.bw_ul_min_mbps
        ul_bw = _clip(ul_bw_inst, self.bw_ul_min_mbps, self.bw_ul_max_mbps)
        cs.net_bw_ul_mbps_est = self._ewma(cs.net_bw_ul_mbps_est or ul_bw, ul_bw)
        cs.net_bw_ul_mbps_est = _clip(cs.net_bw_ul_mbps_est, self.bw_ul_min_mbps, self.bw_ul_max_mbps)

        # 当前系统没有真实下行观测，用 UL 的 EWMA 作为同一链路条件下的保守 proxy。
        dl_bw_inst = max(self.bw_dl_min_mbps, cs.net_bw_ul_mbps_est)
        dl_bw = _clip(dl_bw_inst, self.bw_dl_min_mbps, self.bw_dl_max_mbps)
        cs.net_bw_dl_mbps_est = self._ewma(cs.net_bw_dl_mbps_est or dl_bw, dl_bw)
        cs.net_bw_dl_mbps_est = _clip(cs.net_bw_dl_mbps_est, self.bw_dl_min_mbps, self.bw_dl_max_mbps)

        rtt_inst = self.rtt_floor_ms + self.rtt_upload_coeff_ms * ud
        cs.net_rtt_ms_est = _clip(self._ewma(cs.net_rtt_ms_est or rtt_inst, rtt_inst), 1.0, 2000.0)

        loss_logit = (
            self.loss_bias
            + self.loss_rtt_coeff * (cs.net_rtt_ms_est / 100.0)
            + self.loss_bw_coeff * (self.loss_bw_ref_mbps / max(cs.net_bw_ul_mbps_est, 1e-9))
        )
        loss_inst = _clip(self._sigmoid(loss_logit), 0.0, self.loss_cap)
        cs.net_loss_est = _clip(self._ewma(cs.net_loss_est, loss_inst), 0.0, self.loss_cap)

        # 3) Energy -> SOC
        cpu_power = self.kappa_cpu * cs.cpu_util_est * (cpu_freq_ghz_est ** 3)
        ul_rate_mb_s = max(cs.net_bw_ul_mbps_est / 8.0, 1e-9)
        dl_rate_mb_s = max(cs.net_bw_dl_mbps_est / 8.0, 1e-9)
        ul_time_s = ul_mb / ul_rate_mb_s if ul_mb > 0 else 0.0
        dl_time_s = dl_mb / dl_rate_mb_s if dl_mb > 0 else 0.0
        e_cpu = cpu_power * ct
        e_net = self.tx_power_w * ul_time_s + self.rx_power_w * dl_time_s
        e_base = self.base_power_w * max(ct + ud, self.dt_s)
        soc_drop = self._soc_drop_from_energy(e_cpu + e_net + e_base)
        cs.battery_soc_est = _clip(cs.battery_soc_est - soc_drop, 0.0, 1.0)

        # 4) Thermal RC
        elapsed_s = max(ct + ud, self.dt_s)
        modem_power_proxy = (
            self.tx_power_w * min(ul_time_s, elapsed_s) / max(elapsed_s, 1e-9)
            + self.rx_power_w * min(dl_time_s, elapsed_s) / max(elapsed_s, 1e-9)
        )
        temp_next = (
            float(cs.temp_c_est)
            + (elapsed_s / max(self.thermal_tau_s, 1e-9)) * (self.ambient_temp_c - float(cs.temp_c_est))
            + self.thermal_eta * (cpu_power + modem_power_proxy)
        )
        cs.temp_c_est = _clip(temp_next, self.ambient_temp_c, self.max_temp_c)

        # 5) Memory utilization proxy
        mem_inst = self.mem_base + self.mem_cpu_coeff * cs.cpu_util_est + self.mem_batch_coeff * (
            float(ns) / max(self.mem_sample_ref, 1e-9)
        )
        cs.mem_util_est = _clip(self._ewma(cs.mem_util_est, mem_inst), 0.0, 1.0)
