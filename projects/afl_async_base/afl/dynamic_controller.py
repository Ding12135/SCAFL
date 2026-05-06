from __future__ import annotations

from dataclasses import dataclass

from .runtime_state import SystemState


def _clip_int(x: int, lo: int, hi: int) -> int:
    """Clip integer x into [lo, hi]."""
    return max(lo, min(hi, x))


@dataclass
class DynamicControlOutput:
    """
    Dynamic controller output for first-innovation-point Step2.
    Note: This is currently *observational* only; it does not take over any aggregation logic yet.
    """

    tau_max_t: int
    buffer_target_t: int

    delay_score: float
    heter_score: float
    staleness_score: float


class DynamicController:
    """
    Baseline 线性动态控制：
    - 将 SystemState 归一化为 delay / heterogeneity / staleness 三项 score
    - tau：delay/heter 升高时放宽（+），平均 buffer 陈旧度升高时收紧（−）
    - buffer_target：三项升高时减小目标批量（−），再 clip 到配置区间
    """

    def __init__(self, cfg: dict):
        dc = cfg.get("dynamic_control", {}) if isinstance(cfg, dict) else {}

        self.enabled = bool(dc.get("enabled", True))

        # tau (time/budget) control
        self.tau_min = int(dc.get("tau_min", 1))
        self.tau_base = int(dc.get("tau_base", 8))
        self.tau_max_limit = int(dc.get("tau_max_limit", 32))

        # buffer target control
        self.buffer_min = int(dc.get("buffer_min", 1))
        self.buffer_base = int(dc.get("buffer_base", 8))
        self.buffer_max = int(dc.get("buffer_max", 32))

        # normalization refs
        self.delay_ref = float(dc.get("delay_ref", 1.0))
        self.heter_ref = float(dc.get("heter_ref", 1.0))
        self.staleness_ref = float(dc.get("staleness_ref", 1.0))

        # linear coefficients for tau
        self.tau_coeff_delay = float(dc.get("tau_coeff_delay", 0.5))
        self.tau_coeff_heter = float(dc.get("tau_coeff_heter", 0.5))
        self.tau_coeff_staleness = float(dc.get("tau_coeff_staleness", 0.5))

        # linear coefficients for buffer
        self.buffer_coeff_delay = float(dc.get("buffer_coeff_delay", 0.5))
        self.buffer_coeff_heter = float(dc.get("buffer_coeff_heter", 0.5))
        self.buffer_coeff_staleness = float(dc.get("buffer_coeff_staleness", 0.5))

    def compute(self, state: SystemState) -> DynamicControlOutput:
        # Avoid division by zero by applying a tiny epsilon.
        eps = 1e-12
        delay_ref = max(abs(self.delay_ref), eps)
        heter_ref = max(abs(self.heter_ref), eps)
        staleness_ref = max(abs(self.staleness_ref), eps)

        delay_score = float(state.avg_upload_delay) / delay_ref
        heter_score = float(state.compute_heterogeneity) / heter_ref
        staleness_score = float(state.avg_buffer_staleness) / staleness_ref

        if not self.enabled:
            # When disabled, report baseline targets (still return scores for logging consistency).
            tau_raw = float(self.tau_base)
            buffer_raw = float(self.buffer_base)
        else:
            # Baseline：上传延迟与异构升高时放宽 tau，缓冲平均陈旧度升高时收紧 tau；
            # buffer_target 在拥塞/陈旧时变小以少等待。
            tau_raw = (
                float(self.tau_base)
                + self.tau_coeff_delay * delay_score
                + self.tau_coeff_heter * heter_score
                - self.tau_coeff_staleness * staleness_score
            )
            buffer_raw = (
                float(self.buffer_base)
                - self.buffer_coeff_delay * delay_score
                - self.buffer_coeff_heter * heter_score
                - self.buffer_coeff_staleness * staleness_score
            )

        tau_max_t = _clip_int(int(round(tau_raw)), self.tau_min, self.tau_max_limit)
        buffer_target_t = _clip_int(
            int(round(buffer_raw)), self.buffer_min, self.buffer_max
        )

        return DynamicControlOutput(
            tau_max_t=tau_max_t,
            buffer_target_t=buffer_target_t,
            delay_score=delay_score,
            heter_score=heter_score,
            staleness_score=staleness_score,
        )

