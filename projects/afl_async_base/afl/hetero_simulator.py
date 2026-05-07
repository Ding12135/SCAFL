"""
端侧异构环境模拟（仅影响时延与到达时刻，不作为控制器/policy 的显式输入）。

画像与延迟系数只用于 compute/upload sleep，不写入 UpdateMsg 的“端侧状态”字段。
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, Tuple


def _rng_unit(client_id: int, update_idx: int, cfg: dict, salt: str) -> float:
    """[0,1) 上的稳定伪随机数，与 seed、client、轮次绑定。"""
    base = int(cfg.get("seed", 0))
    raw = f"{base}:{client_id}:{update_idx}:{salt}".encode("utf-8")
    h = hashlib.sha256(raw).digest()
    return struct.unpack("!I", h[:4])[0] / float(2**32)


def _lerp(lo: float, hi: float, u: float) -> float:
    return float(lo) + float(u) * (float(hi) - float(lo))


def get_client_hetero_profile(client_id: int, cfg: dict) -> dict:
    """
    返回每个客户端稳定的异构画像（仅用于产生延迟，不作为控制器输入）。
    """
    _ = cfg
    cid = int(client_id) % 10
    if cid in (0, 1):
        tier = "straggler"
        c_lo, c_hi = 0.18, 0.40
        u_lo, u_hi = 0.12, 0.30
    elif cid in (2, 3, 4):
        tier = "slow"
        c_lo, c_hi = 0.08, 0.18
        u_lo, u_hi = 0.06, 0.12
    elif cid in (5, 6, 7):
        tier = "normal"
        c_lo, c_hi = 0.03, 0.08
        u_lo, u_hi = 0.02, 0.06
    else:
        tier = "fast"
        c_lo, c_hi = 0.00, 0.03
        u_lo, u_hi = 0.00, 0.02
    return {
        "tier": tier,
        "compute_extra_lo": c_lo,
        "compute_extra_hi": c_hi,
        "upload_extra_lo": u_lo,
        "upload_extra_hi": u_hi,
    }


def simulate_compute_delay(client_id: int, update_idx: int, cfg: dict) -> float:
    """该客户端本轮本地训练前的额外 wall-time 延迟（秒）。"""
    het = cfg.get("heterogeneity") if isinstance(cfg, dict) else None
    if not isinstance(het, dict) or not bool(het.get("enabled", False)):
        return 0.0
    if not bool(het.get("compute_delay_enabled", True)):
        return 0.0
    prof = get_client_hetero_profile(client_id, cfg)
    u = _rng_unit(client_id, update_idx, cfg, "compute_extra")
    return max(0.0, _lerp(prof["compute_extra_lo"], prof["compute_extra_hi"], u))


def simulate_upload_delay(client_id: int, update_idx: int, cfg: dict) -> float:
    """该客户端本轮上传前的额外 wall-time 延迟（秒）。"""
    het = cfg.get("heterogeneity") if isinstance(cfg, dict) else None
    if not isinstance(het, dict) or not bool(het.get("enabled", False)):
        return 0.0
    if not bool(het.get("upload_delay_enabled", True)):
        return 0.0
    prof = get_client_hetero_profile(client_id, cfg)
    u = _rng_unit(client_id, update_idx, cfg, "upload_extra")
    return max(0.0, _lerp(prof["upload_extra_lo"], prof["upload_extra_hi"], u))


def hetero_enabled(cfg: dict) -> bool:
    het = cfg.get("heterogeneity") if isinstance(cfg, dict) else None
    return isinstance(het, dict) and bool(het.get("enabled", False))
