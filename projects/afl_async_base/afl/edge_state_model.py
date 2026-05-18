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
    staleness: int
    num_samples: int


class EdgeStateModel:
    """
    论文常见的端侧状态近似建模（server-side proxy）：
    - CPU 利用率：compute_time 归一化 + EWMA
    - 网络状态：由 payload/upload_delay 估计 UL/DL 带宽，并递推 RTT/loss
    - 电池状态：CPU + 网络功耗积分更新 SOC
    - 温度：一阶 RC 热模型
    - 内存：负载压力 proxy（CPU + staleness）映射
    """

    def __init__(self, cfg: dict):
        mc = cfg.get("edge_state_model", {}) if isinstance(cfg, dict) else {}

        self.enabled = bool(mc.get("enabled", True))
        self.dt_s = float(mc.get("dt_s", 1.0))
        self.ewma_alpha = float(mc.get("ewma_alpha", 0.85))

        # CPU proxy
        self.compute_time_ref_s = float(mc.get("compute_time_ref_s", 1.0))
        self.cpu_floor = float(mc.get("cpu_floor", 0.02))

        # Network proxy
        self.uplink_mb_per_sample = float(mc.get("uplink_mb_per_sample", 0.003))
        self.downlink_ratio = float(mc.get("downlink_ratio", 0.12))
        self.min_delay_s = float(mc.get("min_delay_s", 1e-3))
        self.bw_ul_min_mbps = float(mc.get("bw_ul_min_mbps", 0.2))
        self.bw_ul_max_mbps = float(mc.get("bw_ul_max_mbps", 300.0))
        self.bw_dl_min_mbps = float(mc.get("bw_dl_min_mbps", 0.2))
        self.bw_dl_max_mbps = float(mc.get("bw_dl_max_mbps", 600.0))
        self.rtt_floor_ms = float(mc.get("rtt_floor_ms", 8.0))
        self.rtt_upload_coeff_ms = float(mc.get("rtt_upload_coeff_ms", 120.0))
        self.rtt_staleness_coeff_ms = float(mc.get("rtt_staleness_coeff_ms", 1.0))

        # loss logistic
        self.loss_bias = float(mc.get("loss_bias", -3.0))
        self.loss_rtt_coeff = float(mc.get("loss_rtt_coeff", 1.1))
        self.loss_staleness_coeff = float(mc.get("loss_staleness_coeff", 0.9))
        self.loss_staleness_ref = float(mc.get("loss_staleness_ref", 8.0))
        self.loss_cap = float(mc.get("loss_cap", 0.6))

        # Energy / battery
        self.cpu_freq_ghz = float(mc.get("cpu_freq_ghz", 1.8))
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
        self.mem_staleness_coeff = float(mc.get("mem_staleness_coeff", 0.20))
        self.mem_staleness_ref = float(mc.get("mem_staleness_ref", 8.0))

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
        st = max(0, int(obs.staleness))
        ns = max(0, int(obs.num_samples))

        # 1) CPU utilization proxy
        cpu_inst = _clip(ct / max(self.compute_time_ref_s, 1e-9), self.cpu_floor, 1.0)
        cs.cpu_util_est = _clip(self._ewma(cs.cpu_util_est, cpu_inst), 0.0, 1.0)

        # 2) Network state proxy from estimated payload/delay
        ul_mb = float(ns) * max(self.uplink_mb_per_sample, 0.0)
        ul_bw_inst = 8.0 * ul_mb / max(ud, self.min_delay_s) if ul_mb > 0 else self.bw_ul_min_mbps
        ul_bw = _clip(ul_bw_inst, self.bw_ul_min_mbps, self.bw_ul_max_mbps)
        cs.net_bw_ul_mbps_est = self._ewma(cs.net_bw_ul_mbps_est or ul_bw, ul_bw)
        cs.net_bw_ul_mbps_est = _clip(cs.net_bw_ul_mbps_est, self.bw_ul_min_mbps, self.bw_ul_max_mbps)

        dl_mb = ul_mb * max(self.downlink_ratio, 0.0)
        dl_bw_inst = (
            8.0 * dl_mb / max(ud * self.downlink_ratio, self.min_delay_s)
            if dl_mb > 0
            else max(self.bw_dl_min_mbps, cs.net_bw_ul_mbps_est)
        )
        dl_bw = _clip(dl_bw_inst, self.bw_dl_min_mbps, self.bw_dl_max_mbps)
        cs.net_bw_dl_mbps_est = self._ewma(cs.net_bw_dl_mbps_est or dl_bw, dl_bw)
        cs.net_bw_dl_mbps_est = _clip(cs.net_bw_dl_mbps_est, self.bw_dl_min_mbps, self.bw_dl_max_mbps)

        rtt_inst = self.rtt_floor_ms + self.rtt_upload_coeff_ms * ud + self.rtt_staleness_coeff_ms * st
        cs.net_rtt_ms_est = _clip(self._ewma(cs.net_rtt_ms_est or rtt_inst, rtt_inst), 1.0, 2000.0)

        loss_logit = (
            self.loss_bias
            + self.loss_rtt_coeff * (cs.net_rtt_ms_est / 100.0)
            + self.loss_staleness_coeff * (float(st) / max(self.loss_staleness_ref, 1e-9))
        )
        loss_inst = _clip(self._sigmoid(loss_logit), 0.0, self.loss_cap)
        cs.net_loss_est = _clip(self._ewma(cs.net_loss_est, loss_inst), 0.0, self.loss_cap)

        # 3) Energy -> SOC
        cpu_power = self.kappa_cpu * cs.cpu_util_est * (self.cpu_freq_ghz ** 3)
        ul_rate_mb_s = max(cs.net_bw_ul_mbps_est / 8.0, 1e-9)
        dl_rate_mb_s = max(cs.net_bw_dl_mbps_est / 8.0, 1e-9)
        ul_time_s = ul_mb / ul_rate_mb_s if ul_mb > 0 else 0.0
        dl_time_s = dl_mb / dl_rate_mb_s if dl_mb > 0 else 0.0
        e_cpu = cpu_power * self.dt_s
        e_net = self.tx_power_w * ul_time_s + self.rx_power_w * dl_time_s
        e_base = self.base_power_w * self.dt_s
        soc_drop = self._soc_drop_from_energy(e_cpu + e_net + e_base)
        cs.battery_soc_est = _clip(cs.battery_soc_est - soc_drop, 0.0, 1.0)

        # 4) Thermal RC
        modem_power_proxy = (
            self.tx_power_w * min(ul_time_s, self.dt_s) / max(self.dt_s, 1e-9)
            + self.rx_power_w * min(dl_time_s, self.dt_s) / max(self.dt_s, 1e-9)
        )
        temp_next = (
            float(cs.temp_c_est)
            + (self.dt_s / max(self.thermal_tau_s, 1e-9)) * (self.ambient_temp_c - float(cs.temp_c_est))
            + self.thermal_eta * (cpu_power + modem_power_proxy)
        )
        cs.temp_c_est = _clip(temp_next, self.ambient_temp_c, self.max_temp_c)

        # 5) Memory utilization proxy
        mem_inst = self.mem_base + self.mem_cpu_coeff * cs.cpu_util_est + self.mem_staleness_coeff * (
            float(st) / max(self.mem_staleness_ref, 1e-9)
        )
        cs.mem_util_est = _clip(self._ewma(cs.mem_util_est, mem_inst), 0.0, 1.0)
