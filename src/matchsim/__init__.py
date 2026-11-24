"""matchsim: a ranked-matchmaking simulator for comparing rating systems.

Everything here runs on simulated players. Nothing in this package uses
or claims to describe real player data.
"""

from matchsim.config import SimConfig
from matchsim.ratings import Elo, EloDecayingK, Glicko2, RatingSystem
from matchsim.simulation import SimulationResult, run_simulation
from matchsim.replicate import run_replicates, summarise_replicates

__all__ = [
    "SimConfig",
    "RatingSystem",
    "Elo",
    "EloDecayingK",
    "Glicko2",
    "SimulationResult",
    "run_simulation",
    "run_replicates",
    "summarise_replicates",
]
