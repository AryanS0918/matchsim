"""Elo with a fixed K-factor, and Elo with a K-factor that decays with games played."""

from __future__ import annotations

import numpy as np

from matchsim.ratings.base import RatingSystem


class Elo(RatingSystem):
    """Classic Elo.

    Update: r_A <- r_A + K * (S_A - E_A), where E_A = 1 / (1 + 10^((r_B - r_A)/scale)).
    With a single fixed K the update is zero-sum, so the mean rating never moves.
    """

    def __init__(self, k: float = 32.0, initial: float = 1500.0, scale: float = 400.0,
                 name: str | None = None):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = float(k)
        self.initial = float(initial)
        self.scale = float(scale)
        self.name = name or f"elo_k{k:g}"
        self._rating: np.ndarray | None = None
        self.games: np.ndarray | None = None

    # --- state -----------------------------------------------------------
    def reset(self, n_players: int) -> None:
        self._rating = np.full(n_players, self.initial, dtype=float)
        self.games = np.zeros(n_players, dtype=np.int64)

    @property
    def ratings(self) -> np.ndarray:
        if self._rating is None:
            raise RuntimeError("call reset() before use")
        return self._rating

    # --- model -----------------------------------------------------------
    def k_factor(self, games: np.ndarray) -> np.ndarray:
        """K for players who have played ``games`` matches so far."""
        return np.full(np.shape(games), self.k, dtype=float)

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r = self.ratings
        return 1.0 / (1.0 + 10.0 ** ((r[b] - r[a]) / self.scale))

    def update(self, a: np.ndarray, b: np.ndarray, score_a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        score_a = np.asarray(score_a, dtype=float)
        if a.size == 0:
            return
        self._check_disjoint(a, b)

        e_a = self.predict(a, b)
        k_a = self.k_factor(self.games[a])
        k_b = self.k_factor(self.games[b])
        delta = score_a - e_a  # B's surprise is exactly -delta
        self._rating[a] += k_a * delta
        self._rating[b] -= k_b * delta
        self.games[a] += 1
        self.games[b] += 1


class EloDecayingK(Elo):
    """Elo whose K shrinks as a player accumulates games.

    K(n) = k_min + (k_max - k_min) * exp(-n / tau)

    New accounts move fast, established ones settle. Because two opponents can
    have different K, updates are no longer exactly zero-sum, so the rating
    mean can drift; the simulation reports a centred RMSE for that reason.
    """

    def __init__(self, k_max: float = 64.0, k_min: float = 16.0, tau: float = 30.0,
                 initial: float = 1500.0, scale: float = 400.0, name: str | None = None):
        if not 0 < k_min <= k_max:
            raise ValueError("need 0 < k_min <= k_max")
        if tau <= 0:
            raise ValueError("tau must be positive")
        super().__init__(k=k_max, initial=initial, scale=scale,
                         name=name or f"elo_decay_{k_max:g}to{k_min:g}_tau{tau:g}")
        self.k_max = float(k_max)
        self.k_min = float(k_min)
        self.tau = float(tau)

    def k_factor(self, games: np.ndarray) -> np.ndarray:
        games = np.asarray(games, dtype=float)
        return self.k_min + (self.k_max - self.k_min) * np.exp(-games / self.tau)
