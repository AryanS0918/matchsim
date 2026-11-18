import numpy as np
import pytest

from matchsim.ratings import Elo, EloDecayingK


def test_textbook_update():
    elo = Elo(k=32)
    elo.reset(2)
    elo.update(np.array([0]), np.array([1]), np.array([1.0]))
    assert elo.ratings[0] == pytest.approx(1516.0)
    assert elo.ratings[1] == pytest.approx(1484.0)


def test_expected_score_properties():
    elo = Elo()
    elo.reset(3)
    elo.ratings[:] = [1500, 1700, 1300]
    p_ab = elo.predict(np.array([0]), np.array([1]))
    p_ba = elo.predict(np.array([1]), np.array([0]))
    assert p_ab + p_ba == pytest.approx(1.0)
    assert elo.predict(np.array([0]), np.array([0]))[0] == pytest.approx(0.5)
    # 400-point gap -> 10:1 odds
    elo.ratings[:] = [1900, 1500, 0]
    assert elo.predict(np.array([0]), np.array([1]))[0] == pytest.approx(10 / 11)


def test_fixed_k_is_zero_sum():
    rng = np.random.default_rng(1)
    elo = Elo(k=24)
    elo.reset(100)
    for _ in range(50):
        perm = rng.permutation(100)
        a, b = perm[:50], perm[50:]
        elo.update(a, b, (rng.random(50) < 0.5).astype(float))
    assert elo.ratings.mean() == pytest.approx(1500.0)


def test_upset_moves_more_than_expected_win():
    elo = Elo(k=32)
    elo.reset(4)
    elo.ratings[:] = [1800, 1400, 1800, 1400]
    elo.update(np.array([0, 3]), np.array([1, 2]), np.array([1.0, 1.0]))
    favourite_gain = elo.ratings[0] - 1800
    underdog_gain = elo.ratings[3] - 1400
    assert 0 < favourite_gain < underdog_gain


def test_rejects_player_in_two_matches():
    elo = Elo()
    elo.reset(3)
    with pytest.raises(ValueError):
        elo.update(np.array([0, 0]), np.array([1, 2]), np.array([1.0, 0.0]))


def test_decaying_k_schedule():
    sys = EloDecayingK(k_max=64, k_min=16, tau=30)
    k = sys.k_factor(np.arange(0, 500))
    assert k[0] == pytest.approx(64)
    assert np.all(np.diff(k) < 0)
    assert k[-1] == pytest.approx(16, abs=1e-3)


def test_decaying_k_new_player_moves_more():
    sys = EloDecayingK(k_max=64, k_min=16, tau=30)
    sys.reset(2)
    sys.games[1] = 300  # veteran
    sys.update(np.array([0]), np.array([1]), np.array([1.0]))
    assert sys.ratings[0] - 1500 > 1500 - sys.ratings[1]
