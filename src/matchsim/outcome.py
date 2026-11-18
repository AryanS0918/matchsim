"""Ground-truth match outcome model (uses hidden true skill, never ratings)."""

from __future__ import annotations

import numpy as np

from matchsim.config import OutcomeMode


def win_probability(skill_a, skill_b, scale: float = 400.0):
    """P(A beats B) under a logistic model on the skill gap (Elo scale)."""
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(skill_b) - np.asarray(skill_a)) / scale))


def sample_scores(p_win: np.ndarray, rng: np.random.Generator, mode: OutcomeMode) -> np.ndarray:
    """Turn win probabilities into scores for player A (1 = win, 0 = loss).

    In "expected" mode the score is the probability itself, which removes
    outcome noise entirely.
    """
    if mode == "expected":
        return np.asarray(p_win, dtype=float).copy()
    return (rng.random(np.shape(p_win)) < p_win).astype(float)
