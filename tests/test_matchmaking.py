import numpy as np

from matchsim.matchmaking import WideningWindow, pair_players


def test_window_widens_and_caps():
    w = WideningWindow(base=50, growth=25, max=200)
    assert list(w(np.array([0, 2, 100]))) == [50, 100, 200]


def test_pairs_are_disjoint_and_from_queue():
    rng = np.random.default_rng(0)
    ratings = rng.normal(1500, 300, 1000)
    waits = rng.integers(0, 10, 1000)
    queued = rng.choice(1000, 400, replace=False)
    a, b = pair_players(queued, ratings, waits, WideningWindow(), rng)
    both = np.concatenate([a, b])
    assert len(a) == len(b) > 0
    assert np.unique(both).size == both.size
    assert set(both) <= set(queued)


def test_respects_window():
    rng = np.random.default_rng(0)
    w = WideningWindow(base=50, growth=25, max=600)
    ratings = rng.normal(1500, 300, 500)
    waits = rng.integers(0, 5, 500)
    a, b = pair_players(np.arange(500), ratings, waits, w, rng)
    gap = np.abs(ratings[a] - ratings[b])
    allowed = np.maximum(w(waits[a]), w(waits[b]))
    assert np.all(gap <= allowed + 1e-9)


def test_waiting_unlocks_a_match():
    rng = np.random.default_rng(0)
    w = WideningWindow(base=50, growth=25, max=600)
    ratings = np.array([1500.0, 1600.0])  # 100-point gap
    a, _ = pair_players(np.array([0, 1]), ratings, np.array([0, 0]), w, rng)
    assert a.size == 0                    # window 50: no match
    a, _ = pair_players(np.array([0, 1]), ratings, np.array([2, 0]), w, rng)
    assert a.size == 1                    # one player waited 2 ticks -> window 100


def test_tiny_queue():
    rng = np.random.default_rng(0)
    a, b = pair_players(np.array([3]), np.zeros(5), np.zeros(5), WideningWindow(), rng)
    assert a.size == b.size == 0
