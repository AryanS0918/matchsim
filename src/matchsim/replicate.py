"""Replicated runs and confidence intervals.

In this experiment the unit of replication is a whole simulation run (one
seed), not a match. Matches inside a run share a population and a rating
history, so they are not independent; treating them as i.i.d. would give
CIs that are far too narrow.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from matchsim.config import SimConfig
from matchsim.ratings.base import RatingSystem
from matchsim.simulation import run_simulation


def run_replicates(
    arms: dict[str, Callable[[], RatingSystem]],
    config: SimConfig,
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every arm on every seed (same seed across arms = paired design).

    Returns (per-run summaries, stacked histories).
    """
    summaries, histories = [], []
    for seed in seeds:
        cfg = config.with_(seed=seed)
        for arm, make_system in arms.items():
            res = run_simulation(make_system(), cfg)
            s = res.summary()
            s["arm"] = arm
            summaries.append(s)
            h = res.history.copy()
            h["arm"], h["seed"] = arm, seed
            histories.append(h)
    return pd.DataFrame(summaries), pd.concat(histories, ignore_index=True)


def mean_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-based confidence interval."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    m = float(x.mean())
    if x.size < 2:
        return m, float("nan"), float("nan")
    half = stats.t.ppf(0.5 + level / 2, df=x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return m, float(m - half), float(m + half)


def summarise_replicates(summaries: pd.DataFrame,
                         metrics: tuple[str, ...] = ("final_rmse_centred", "final_spearman",
                                                     "mean_unfairness", "median_games_to_settle"),
                         level: float = 0.95) -> pd.DataFrame:
    rows = []
    for arm, g in summaries.groupby("arm", sort=False):
        for m in metrics:
            mean, lo, hi = mean_ci(g[m], level)
            rows.append({"arm": arm, "metric": m, "mean": mean, "ci_low": lo,
                         "ci_high": hi, "n_runs": int(g[m].notna().sum())})
    return pd.DataFrame(rows)


def paired_difference(summaries: pd.DataFrame, metric: str, arm_a: str, arm_b: str,
                      level: float = 0.95) -> dict:
    """Mean of (arm_a - arm_b) across seeds, with a CI. Uses the pairing by seed."""
    wide = summaries.pivot(index="seed", columns="arm", values=metric)
    diff = (wide[arm_a] - wide[arm_b]).to_numpy()
    mean, lo, hi = mean_ci(diff, level)
    return {"metric": metric, "contrast": f"{arm_a} - {arm_b}", "mean_diff": mean,
            "ci_low": lo, "ci_high": hi, "n_seeds": int(np.isfinite(diff).sum())}
