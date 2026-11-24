import numpy as np
import pandas as pd

from matchsim import Elo, EloDecayingK, SimConfig, run_simulation
from matchsim.metrics import centred_rmse, spearman, unfairness
from matchsim.replicate import paired_difference, run_replicates

SMALL = SimConfig(n_players=400, n_ticks=200, record_every=20, seed=7)


def test_noiseless_elo_recovers_true_skill():
    """Sanity check from the brief: with no outcome noise, Elo converges to true skill.

    In "expected" mode the score equals the true win probability, so the only
    fixed point of the Elo update is rating = true skill (up to a common offset).
    Random pairing isolates the rating algorithm from matchmaking.
    """
    cfg = SMALL.with_(outcome_mode="expected", matchmaking="random", n_ticks=2000)
    res = run_simulation(Elo(k=32), cfg)
    start = res.history["rmse_centred"].iloc[0]
    end = res.history["rmse_centred"].iloc[-1]
    assert start > 200          # everyone starts at 1500, skill sd = 250
    assert end < 5              # essentially exact
    assert res.history["spearman"].iloc[-1] > 0.999


def test_noisy_elo_learns_but_has_a_noise_floor():
    cfg = SMALL.with_(matchmaking="random", n_ticks=600)
    h = run_simulation(Elo(k=32), cfg).history
    assert h["rmse_centred"].iloc[-1] < 0.4 * h["rmse_centred"].iloc[0]
    assert h["spearman"].iloc[-1] > 0.9
    # with outcome noise, fixed-K Elo never settles: a non-zero error floor remains
    late = h.loc[h["tick"] >= 300, "rmse_centred"]
    assert late.min() > 20


def test_nearest_matchmaking_compresses_rating_scale():
    """Known behaviour, pinned so it can't change silently.

    With noiseless outcomes and closest-rating pairing, Elo orders players
    almost perfectly but the rating *spread* stays far too narrow: players are
    only ever compared with neighbours, so scale information spreads very slowly.
    """
    cfg = SMALL.with_(outcome_mode="expected", n_ticks=600)
    res = run_simulation(Elo(k=32), cfg)
    assert res.history["spearman"].iloc[-1] > 0.99
    assert res.final_ratings.std() < 0.3 * res.true_skill.std()


def test_same_seed_same_result():
    r1 = run_simulation(Elo(k=32), SMALL)
    r2 = run_simulation(Elo(k=32), SMALL)
    pd.testing.assert_frame_equal(r1.history, r2.history)
    np.testing.assert_array_equal(r1.final_ratings, r2.final_ratings)


def test_different_seed_different_result():
    r1 = run_simulation(Elo(k=32), SMALL)
    r2 = run_simulation(Elo(k=32), SMALL.with_(seed=8))
    assert not np.allclose(r1.final_ratings, r2.final_ratings)


def test_arms_share_population_under_same_seed():
    r1 = run_simulation(Elo(k=32), SMALL)
    r2 = run_simulation(EloDecayingK(), SMALL)
    np.testing.assert_array_equal(r1.true_skill, r2.true_skill)


def test_history_is_well_formed():
    res = run_simulation(Elo(), SMALL, record_matches=True)
    h = res.history
    assert h["tick"].is_monotonic_increasing and h["tick"].iloc[-1] == SMALL.n_ticks
    assert h["games_per_player"].is_monotonic_increasing
    assert res.matches is not None and len(res.matches) == res.games.sum() // 2
    assert res.matches["p_true"].between(0, 1).all()


def test_observable_match_metrics_are_reported():
    res = run_simulation(Elo(), SMALL, record_matches=True)
    assert res.matches is not None
    for metric in ("prediction_log_loss", "favorite_win_rate", "mean_queue_time"):
        assert metric in res.history
        assert np.isfinite(res.summary()[metric])
    assert res.summary()["prediction_log_loss"] >= 0
    assert 0 <= res.summary()["favorite_win_rate"] <= 1
    assert res.summary()["mean_queue_time"] >= 0


def test_settling_time_bookkeeping():
    cfg = SMALL.with_(outcome_mode="expected", matchmaking="random", n_ticks=2000)
    res = run_simulation(Elo(k=32), cfg)
    assert res.settled.all()                         # noiseless: everyone settles
    assert (res.settle_game <= res.games).all()
    far = np.abs(res.true_skill - res.true_skill.mean()) > 2 * cfg.converge_eps
    near = np.abs(res.true_skill - res.true_skill.mean()) < 0.5 * cfg.converge_eps
    # players far from the default rating take longer to settle
    assert np.median(res.settle_game[far]) > np.median(res.settle_game[near])


def test_unsettled_players_are_censored_at_games_played():
    res = run_simulation(Elo(k=32), SMALL.with_(n_ticks=20))
    assert (~res.settled).any()
    np.testing.assert_array_equal(res.settle_game[~res.settled], res.games[~res.settled])


def test_metric_helpers():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert centred_rmse(x + 100, x) == 0
    assert spearman(x, x ** 3) == 1
    assert np.isnan(spearman(np.ones(4), x))
    assert unfairness(np.array([0.5, 0.5])) == 0
    assert unfairness(np.array([1.0, 0.0])) == 0.5


def test_kaplan_meier_median_respects_right_censoring():
    from matchsim.metrics import kaplan_meier_median

    assert kaplan_meier_median(np.array([1, 2, 3, 4]),
                               np.array([True, True, False, False])) == 2.0
    assert np.isnan(kaplan_meier_median(
        np.array([1, 2]), np.array([False, False])))


def test_replicates_and_paired_ci():
    cfg = SMALL.with_(n_ticks=60)
    arms = {"fixed": lambda: Elo(k=32), "decay": lambda: EloDecayingK()}
    summ, hist = run_replicates(arms, cfg, seeds=range(3))
    assert len(summ) == 6 and set(hist["arm"]) == {"fixed", "decay"}
    d = paired_difference(summ, "final_rmse_centred", "fixed", "decay")
    assert d["n_seeds"] == 3 and d["ci_low"] <= d["mean_diff"] <= d["ci_high"]
