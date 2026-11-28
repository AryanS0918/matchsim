"""Glicko-2 rating, deviation, and volatility updates."""

from __future__ import annotations

import numpy as np

from matchsim.ratings.base import RatingSystem

_SCALE = 173.7178
class Glicko2(RatingSystem):
    """Glicko-2 with one rating period per simulation tick."""

    name = "glicko2"

    def __init__(
        self,
        initial: float = 1500.0,
        initial_rd: float = 350.0,
        volatility: float = 0.06,
        tau: float = 0.5,
        name: str = "glicko2",
    ):
        if initial_rd <= 0 or volatility <= 0 or tau <= 0:
            raise ValueError("initial_rd, volatility, and tau must be positive")
        self.initial = float(initial)
        self.initial_rd = float(initial_rd)
        self.initial_volatility = float(volatility)
        self.tau = float(tau)
        self.name = name
        self._rating: np.ndarray | None = None
        self.rd: np.ndarray | None = None
        self.volatilities: np.ndarray | None = None

    def reset(self, n_players: int) -> None:
        self._rating = np.full(n_players, self.initial, dtype=float)
        self.rd = np.full(n_players, self.initial_rd, dtype=float)
        self.volatilities = np.full(n_players, self.initial_volatility, dtype=float)

    @property
    def ratings(self) -> np.ndarray:
        if self._rating is None:
            raise RuntimeError("call reset() before use")
        return self._rating

    def predict(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu_a = (self.ratings[a] - 1500.0) / _SCALE
        mu_b = (self.ratings[b] - 1500.0) / _SCALE
        phi_b = self.rd[b] / _SCALE
        g_b = 1.0 / np.sqrt(1.0 + 3.0 * phi_b**2 / np.pi**2)
        return 1.0 / (1.0 + np.exp(-g_b * (mu_a - mu_b)))

    def _updated_volatility(self, phi: float, delta: float, variance: float,
                            sigma: float) -> float:
        a = np.log(sigma**2)

        def f(x: float) -> float:
            ex = np.exp(x)
            numerator = ex * (delta**2 - phi**2 - variance - ex)
            denominator = 2.0 * (phi**2 + variance + ex) ** 2
            return numerator / denominator - (x - a) / self.tau**2

        lower = a
        if delta**2 > phi**2 + variance:
            upper = np.log(delta**2 - phi**2 - variance)
        else:
            k = 1
            upper = a - k * self.tau
            while f(upper) < 0:
                k += 1
                upper = a - k * self.tau

        f_lower, f_upper = f(lower), f(upper)
        while abs(upper - lower) > 1e-6:
            candidate = lower + (lower - upper) * f_lower / (f_upper - f_lower)
            f_candidate = f(candidate)
            if f_candidate * f_upper <= 0:
                lower, f_lower = upper, f_upper
            else:
                f_lower /= 2.0
            upper, f_upper = candidate, f_candidate
        return float(np.exp(lower / 2.0))

    def update(self, a: np.ndarray, b: np.ndarray, score_a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        score_a = np.asarray(score_a, dtype=float)
        if a.size != b.size or a.size != score_a.size:
            raise ValueError("a, b, and score_a must have matching lengths")
        if np.any((score_a < 0.0) | (score_a > 1.0)):
            raise ValueError("scores must be between 0 and 1")
        if a.size:
            players, inverse = np.unique(
                np.concatenate((a, b)), return_inverse=True)
            opponents = np.concatenate((b, a))
            scores = np.concatenate((score_a, 1.0 - score_a))
            mu = (self.ratings[players] - 1500.0) / _SCALE
            phi = self.rd[players] / _SCALE
            sigma = self.volatilities[players]
            opponent_mu = (self.ratings[opponents] - 1500.0) / _SCALE
            opponent_phi = self.rd[opponents] / _SCALE

            g = 1.0 / np.sqrt(1.0 + 3.0 * opponent_phi**2 / np.pi**2)
            game_mu = (self.ratings[np.concatenate((a, b))] - 1500.0) / _SCALE
            expected = 1.0 / (1.0 + np.exp(-g * (game_mu - opponent_mu)))
            information = np.bincount(
                inverse, weights=g**2 * expected * (1.0 - expected))
            variance = 1.0 / information
            performance = np.bincount(inverse, weights=g * (scores - expected))
            delta = variance * performance

            new_sigma = np.array([
                self._updated_volatility(phi_i, delta_i, variance_i, sigma_i)
                for phi_i, delta_i, variance_i, sigma_i
                in zip(phi, delta, variance, sigma)
            ])
            phi_star = np.sqrt(phi**2 + new_sigma**2)
            new_phi = 1.0 / np.sqrt(1.0 / phi_star**2 + 1.0 / variance)
            new_mu = mu + new_phi**2 * performance

            self._rating[players] = 1500.0 + _SCALE * new_mu
            self.rd[players] = _SCALE * new_phi
            self.volatilities[players] = new_sigma

        played = np.zeros(self.ratings.size, dtype=bool)
        played[a] = True
        played[b] = True
        idle = ~played
        idle_phi = np.sqrt(
            (self.rd[idle] / _SCALE) ** 2 + self.volatilities[idle] ** 2
        )
        self.rd[idle] = idle_phi * _SCALE
