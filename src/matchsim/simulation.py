"""The main simulation loop.

Each tick:
  1. idle players join the queue with probability ``queue_prob``
  2. the matchmaker pairs queued players by current rating
  3. outcomes are drawn from *true* skill (the rating system never sees it)
  4. the rating system updates
  5. unmatched players wait one more tick, which widens their window

Randomness comes from separate streams spawned off one seed, so two rating
systems run with the same seed face the identical population and identical
queue-entry draws (common random numbers). That makes arm-vs-arm comparisons
paired, which cuts variance a lot.

Hooks for later steps:
  * smurfs  -> change ``true_skill`` for a subset at t=0 (ratings stay default)
  * tilt    -> make ``effective_skill`` depend on ``loss_streak``
  * churn   -> set ``active[i] = False``; inactive players never queue
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matchsim import metrics
from matchsim.config import SimConfig
from matchsim.matchmaking import WideningWindow, pair_players, pair_random
from matchsim.outcome import sample_scores, win_probability
from matchsim.ratings.base import RatingSystem


@dataclass
class SimulationResult:
    system_name: str
    config: SimConfig
    history: pd.DataFrame
    true_skill: np.ndarray
    final_ratings: np.ndarray
    games: np.ndarray
    settle_game: np.ndarray  # per player: games after which |error| stayed <= converge_eps
    settled: np.ndarray      # False = still outside the band at the end (right-censored)
    matches: pd.DataFrame | None = field(default=None, repr=False)

    def summary(self) -> dict:
        h = self.history.iloc[-1]
        st = self.settle_game[self.settled]
        return {
            "system": self.system_name,
            "seed": self.config.seed,
            "final_rmse_centred": h["rmse_centred"],
            "final_spearman": h["spearman"],
            "mean_unfairness": float(np.average(self.history["unfairness"].dropna(),
                                                weights=self.history.loc[self.history["unfairness"].notna(), "n_matches"])),
            # Naive median over settled players only. Censoring biases it down;
            # use Kaplan-Meier on (settle_game, settled) for the real analysis.
            "median_games_to_settle": float(np.median(st)) if st.size else float("nan"),
            "share_settled": float(self.settled.mean()),
            "games_per_player": float(self.games.mean()),
        }


def run_simulation(system: RatingSystem, config: SimConfig,
                   record_matches: bool = False) -> SimulationResult:
    n = config.n_players
    pop_rng, queue_rng, mm_rng, outcome_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(4)
    )

    true_skill = pop_rng.normal(config.skill_mean, config.skill_sd, size=n)
    system.reset(n)
    window = WideningWindow(config.window_base, config.window_growth, config.window_max)

    in_queue = np.zeros(n, dtype=bool)
    wait = np.zeros(n, dtype=np.int64)
    games = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)          # churn hook
    loss_streak = np.zeros(n, dtype=np.int64)  # tilt / churn hook
    # Settling time: last game count at which a player was outside the band.
    # -1 means "never outside" (default rating was already close enough).
    init_err = np.abs((system.ratings - system.ratings.mean()) - (true_skill - true_skill.mean()))
    last_outside = np.where(init_err > config.converge_eps, 0, -1).astype(np.int64)

    rows: list[dict] = []
    match_log: list[pd.DataFrame] = []
    window_p: list[np.ndarray] = []

    def snapshot(tick: int) -> None:
        r = system.ratings
        p = np.concatenate(window_p) if window_p else np.empty(0)
        rows.append({
            "tick": tick,
            "games_per_player": float(games.mean()),
            "rmse": metrics.rmse(r, true_skill),
            "rmse_centred": metrics.centred_rmse(r, true_skill),
            "spearman": metrics.spearman(r, true_skill),
            "unfairness": metrics.unfairness(p),
            "n_matches": int(p.size),
            "queue_size": int(in_queue.sum()),
            "mean_wait_queued": float(wait[in_queue].mean()) if in_queue.any() else 0.0,
        })
        window_p.clear()

    snapshot(0)
    for tick in range(1, config.n_ticks + 1):
        # 1. queue entry (fixed-size draw keeps this stream aligned across arms)
        joins = queue_rng.random(n) < config.queue_prob
        in_queue |= joins & active & ~in_queue

        # 2. matchmaking
        queued = np.flatnonzero(in_queue)
        if config.matchmaking == "random":
            a, b = pair_random(queued, mm_rng)
        else:
            a, b = pair_players(queued, system.ratings, wait, window, mm_rng)

        if a.size:
            # 3. outcomes from true skill
            eff_a, eff_b = true_skill[a], true_skill[b]  # tilt will modify these
            p_true = win_probability(eff_a, eff_b, config.logistic_scale)
            p_pred = system.predict(a, b)
            score_a = sample_scores(p_true, outcome_rng, config.outcome_mode)

            # 4. rating update
            system.update(a, b, score_a)

            # bookkeeping
            games[a] += 1
            games[b] += 1
            a_won = score_a > 0.5
            winners = np.where(a_won, a, b)
            losers = np.where(a_won, b, a)
            loss_streak[winners] = 0
            loss_streak[losers] += 1

            if record_matches:
                match_log.append(pd.DataFrame({
                    "tick": tick, "a": a, "b": b,
                    "skill_a": eff_a, "skill_b": eff_b,
                    "p_true": p_true, "p_pred": p_pred, "score_a": score_a,
                    "wait_a": wait[a], "wait_b": wait[b],
                }))
            window_p.append(p_true)

            # settling-time bookkeeping (centred error, see metrics.py)
            players = np.concatenate([a, b])
            r = system.ratings
            offset = r.mean() - true_skill.mean()
            err = np.abs(r[players] - offset - true_skill[players])
            out = players[err > config.converge_eps]
            last_outside[out] = games[out]

            in_queue[players] = False
            wait[players] = 0

        # 5. everyone still queued waits longer
        wait[in_queue] += 1

        if tick % config.record_every == 0 or tick == config.n_ticks:
            snapshot(tick)

    r = system.ratings
    final_err = np.abs((r - r.mean()) - (true_skill - true_skill.mean()))
    settled = final_err <= config.converge_eps

    history = pd.DataFrame(rows).drop_duplicates("tick", keep="last").reset_index(drop=True)
    return SimulationResult(
        system_name=system.name,
        config=config,
        history=history,
        true_skill=true_skill,
        final_ratings=system.ratings.copy(),
        games=games,
        settle_game=np.where(settled, last_outside + 1, games),
        settled=settled,
        matches=pd.concat(match_log, ignore_index=True) if match_log else None,
    )
