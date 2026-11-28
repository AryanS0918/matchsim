import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_h7_churn import reconstruct_survival
from matchsim import Elo, SimConfig, run_simulation


def test_churn_reconstruction_is_censored_and_player_level():
    result = run_simulation(
        Elo(k=32),
        SimConfig(n_players=40, n_ticks=40, seed=3),
        record_matches=True,
    )
    for threshold in (3, 5, 8):
        survival = reconstruct_survival(result, threshold)
        assert len(survival) == 40
        assert set(survival["event"].unique()).issubset({0, 1})
        assert survival["duration"].between(1, 40).all()
        assert np.isfinite(survival["true_skill_distance"]).all()
