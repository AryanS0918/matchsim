"""Simulation configuration.

One frozen dataclass holds every knob, so a run is fully described by
(config, rating system, seed). Save it next to any result you report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

OutcomeMode = Literal["sampled", "expected"]


@dataclass(frozen=True)
class SimConfig:
    # Population
    n_players: int = 10_000
    skill_mean: float = 1500.0
    skill_sd: float = 250.0

    # Outcome model: P(A beats B) = 1 / (1 + 10^((s_B - s_A) / logistic_scale)).
    # "sampled"  -> each match is a Bernoulli draw (the realistic, noisy case)
    # "expected" -> the score is the win probability itself (noiseless; used
    #               for the sanity check that Elo recovers true skill)
    logistic_scale: float = 400.0
    outcome_mode: OutcomeMode = "sampled"

    # Time and activity
    n_ticks: int = 500
    queue_prob: float = 0.3  # chance an idle player joins the queue each tick

    # Matchmaking policy:
    # "nearest" -> pair by closest rating within a widening window (the realistic case)
    # "random"  -> pair queued players at random, ignoring rating. Isolates the
    #              rating algorithm from matchmaking (used in the sanity check).
    matchmaking: Literal["nearest", "random"] = "nearest"

    # Matchmaking window (rating points), widening with time spent in queue
    window_base: float = 50.0
    window_growth: float = 25.0  # extra points per tick waited
    window_max: float = 600.0

    # Measurement
    record_every: int = 10  # ticks between history snapshots
    converge_eps: float = 100.0  # |centred rating error| counted as "converged"

    seed: int = 0

    def with_(self, **changes) -> "SimConfig":
        """Return a copy with some fields changed (handy for sensitivity grids)."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.n_players < 2:
            raise ValueError("need at least 2 players")
        if not 0.0 < self.queue_prob <= 1.0:
            raise ValueError("queue_prob must be in (0, 1]")
        if self.outcome_mode not in ("sampled", "expected"):
            raise ValueError(f"unknown outcome_mode {self.outcome_mode!r}")
        if self.matchmaking not in ("nearest", "random"):
            raise ValueError(f"unknown matchmaking {self.matchmaking!r}")
        if self.window_base < 0 or self.window_growth < 0:
            raise ValueError("window parameters must be non-negative")
