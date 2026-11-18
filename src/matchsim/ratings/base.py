"""Interface every rating system implements.

Systems are vectorised: one call to ``update`` processes every match played
in a tick. Matches within a tick are disjoint (no player appears twice), so
a simultaneous update is exact rather than an approximation.

Glicko-2 (step 2) and TrueSkill-lite slot in by subclassing this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RatingSystem(ABC):
    name: str = "base"

    @abstractmethod
    def reset(self, n_players: int) -> None:
        """Initialise state for a fresh population of ``n_players``."""

    @property
    @abstractmethod
    def ratings(self) -> np.ndarray:
        """Current point estimate of every player's skill (used for matchmaking)."""

    @abstractmethod
    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The system's own estimate of P(a beats b)."""

    @abstractmethod
    def update(self, a: np.ndarray, b: np.ndarray, score_a: np.ndarray) -> None:
        """Apply results. ``score_a`` is 1 for an A win, 0 for a loss (or in between)."""

    @staticmethod
    def _check_disjoint(a: np.ndarray, b: np.ndarray) -> None:
        both = np.concatenate([a, b])
        if np.unique(both).size != both.size:
            raise ValueError("a player appears in more than one match this tick")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
