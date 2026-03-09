"""VolStraddle Strategy subpackage — VR compression multi-TF detector.

R35 A1-D50: D_mom8 directional entry on VR < 0.50 across 15m/1h/4h.
R44d V17: MA Cluster Squeeze on 4h (complementary strategy).
OOS validated: N=799, EV=+29.3bp, PF=1.376, t=3.48.
"""

from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import (
    check_ma_cluster,
    compute_momentum,
    compute_sma,
    compute_squeeze_dur,
    compute_vr,
    gen_vr_dur_signal,
    gen_vr_signal,
    make_direction_momentum,
    resample_ohlcv,
)
from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy

__all__ = [
    "VolStraddleConfig",
    "VolStraddleStrategy",
    "check_ma_cluster",
    "compute_momentum",
    "compute_sma",
    "compute_squeeze_dur",
    "compute_vr",
    "gen_vr_dur_signal",
    "gen_vr_signal",
    "make_direction_momentum",
    "resample_ohlcv",
]
