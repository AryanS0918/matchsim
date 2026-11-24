import numpy as np
import pytest

from matchsim.ratings import Glicko2


def test_glickman_worked_example():
    system = Glicko2(initial_rd=200, volatility=0.06)
    system.reset(4)
    system.ratings[:] = [1500, 1400, 1550, 1700]
    system.rd[:] = [200, 30, 100, 300]
    system.update(
        np.array([0, 0, 0]),
        np.array([1, 2, 3]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert system.ratings[0] == pytest.approx(1464.06, abs=0.01)
    assert system.rd[0] == pytest.approx(151.52, abs=0.01)
    assert system.volatilities[0] == pytest.approx(0.05999, abs=1e-5)


def test_idle_rating_deviation_grows_during_empty_period():
    system = Glicko2()
    system.reset(2)
    system.update(np.array([], dtype=int), np.array([], dtype=int),
                  np.array([], dtype=float))
    assert np.all(system.rd > 350)
    np.testing.assert_array_equal(system.ratings, np.full(2, 1500.0))


def test_rejects_scores_outside_probability_range():
    system = Glicko2()
    system.reset(2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        system.update(np.array([0]), np.array([1]), np.array([1.1]))
