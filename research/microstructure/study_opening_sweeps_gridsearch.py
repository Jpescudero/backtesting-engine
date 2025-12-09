"""
Grid-search optimizer for Opening Sweep V2
Evaluates combinations of wick-factor, ATR percentile, and volume percentile.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from study_opening_sweeps_v2 import (
    load_sessions,
    detect_sweep_signals,
    compute_dynamic_forward_returns,
    SweepStudyConfigV2
)

from src.config.paths import REPORTS_DIR, ensure_directories_exist


def run_single(df, atr_norm, wick, atrp, volp):
    cfg = SweepStudyConfigV2(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        timezone="Europe/Madrid",
        min_wick_factor=wick,
        min_atr_percentile=atrp,
        require_volume_percentile=volp,
    )

    signals = detect_sweep_signals(df, cfg)
    entries = signals == 1

    fwd = compute_dynamic_forward_returns(df, atr_norm)
    trade_returns = fwd[entries].dropna()

    if len(trade_returns) < 200:
        return None

    sharpe = trade_returns.mean() / trade_returns.std()
    winrate = (trade_returns > 0).mean()

    return {
        "wick_factor": wick,
        "atr_percentile": atrp,
        "vol_percentile": volp,
        "sharpe": sharpe,
        "winrate": winrate,
        "signals": len(trade_returns),
        "mean": trade_returns.mean(),
    }


def main():
    ensure_directories_exist()

    # Load once
    df, _ = load_sessions("NDXm", range(2018, 2023),
                          windows=(("08:55", "10:00"), ("14:55", "16:00")),
                          tz="Europe/Madrid")

    # Build ATR norm
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
    atr = tr.rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()

    results = []
    grid_wick = [1.2, 1.5, 1.8, 2.0]
    grid_atr  = [0.2, 0.4, 0.6]
    grid_vol  = [0.4, 0.6, 0.8]

    for w in grid_wick:
        for a in grid_atr:
            for v in grid_vol:
                out = run_single(df, atr_norm, w, a, v)
                if out is not None:
                    results.append(out)

    dfres = pd.DataFrame(results)
    outdir = REPORTS_DIR / "research" / "microstructure" / "gridsearch"
    outdir.mkdir(parents=True, exist_ok=True)

    dfres.to_csv(outdir / "gridsearch_results.csv", index=False)
    print(dfres.sort_values("sharpe", ascending=False).head(10))


if __name__ == "__main__":
    main()
