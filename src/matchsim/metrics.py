"""Accuracy and fairness metrics.

A note on centring: Elo-type systems only identify skill *differences*. If the
population's true mean is 1512 and every rating starts at 1500, a perfect
fixed-K Elo still sits 12 points off everyone. ``centred_rmse`` removes that
common offset, and it is the primary accuracy metric. Raw RMSE is kept too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(ratings: np.ndarray, skill: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ratings - skill) ** 2)))


def centred_rmse(ratings: np.ndarray, skill: np.ndarray) -> float:
    err = (ratings - ratings.mean()) - (skill - skill.mean())
    return float(np.sqrt(np.mean(err ** 2)))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=float)
    # group ties
    boundaries = np.flatnonzero(np.diff(sorted_x)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [x.size]])
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e - 1) / 2.0
    return ranks


def spearman(ratings: np.ndarray, skill: np.ndarray) -> float:
    """Spearman rank correlation; NaN when ratings are all tied (e.g. at t=0)."""
    if np.ptp(ratings) == 0 or np.ptp(skill) == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(ratings), _ranks(skill))[0, 1])


def unfairness(p_win: np.ndarray) -> float:
    """Mean |P(A wins) - 0.5|. 0 = perfect coin-flip matches, 0.5 = foregone conclusions."""
    p_win = np.asarray(p_win)
    return float(np.mean(np.abs(p_win - 0.5))) if p_win.size else float("nan")


def games_to_threshold(history: pd.DataFrame, threshold: float,
                       metric: str = "rmse_centred") -> float:
    """Mean games per player at the first snapshot where ``metric`` <= threshold.

    Returns NaN if the threshold is never reached. This is a population-level
    convergence measure; per-player settling times are in
    ``SimulationResult.settle_game`` / ``.settled``.
    """
    hit = history.loc[history[metric] <= threshold]
    return float(hit["games_per_player"].iloc[0]) if len(hit) else float("nan")


def kaplan_meier_median(times: np.ndarray, events: np.ndarray) -> float:
    """Estimate the median event time with right-censored observations."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.ndim != 1 or events.ndim != 1 or times.size != events.size:
        raise ValueError("times and events must be one-dimensional arrays of equal size")
    if np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("times must be finite and non-negative")
    if times.size == 0:
        return float("nan")

    survival = 1.0
    for time in np.unique(times[events]):
        at_risk = np.count_nonzero(times >= time)
        observed_events = np.count_nonzero((times == time) & events)
        survival *= 1.0 - observed_events / at_risk
        if survival <= 0.5:
            return float(time)
    return float("nan")
