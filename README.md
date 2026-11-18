# matchsim — Elo vs Glicko-2 in a simulated ranked queue

> **Status: step 1 of 5** (Elo, the matchmaking loop and tests). Glicko-2,
> smurfs, tilt, churn and the full experiment analysis are still to come.
> Everything in this repository is a **simulation**. No real player data is used.

## Question

Which rating system finds a player's true skill fastest and gives the fairest
matches? How do smurfs and tilt change that, and what does each system do to
retention? The hypotheses are fixed in advance in [`HYPOTHESES.md`](HYPOTHESES.md).

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # 23 tests, about 2 s
python scripts/run_baseline.py --seeds 10   # about 45 s at 10k players
```

```python
from matchsim import Elo, SimConfig, run_simulation

res = run_simulation(Elo(k=32), SimConfig(n_players=10_000, n_ticks=500, seed=0))
res.history        # RMSE, Spearman, unfairness and queue stats over time
res.summary()
```

## Layout

```
src/matchsim/
  config.py        SimConfig: every knob in one frozen dataclass
  outcome.py       ground truth: P(win) from the true-skill gap
  ratings/         RatingSystem interface, Elo, EloDecayingK (Glicko-2 goes here)
  matchmaking.py   closest-rating pairing with a widening queue window; random pairing
  simulation.py    the tick loop, metrics history, per-player settling times
  metrics.py       RMSE (raw and centred), Spearman, unfairness
  replicate.py     multi-seed runs, t-intervals, paired differences
scripts/run_baseline.py
tests/
HYPOTHESES.md
```

## Model

- **Players:** true skill ~ N(1500, 250). The rating systems never see it.
- **Outcomes:** P(A beats B) = 1 / (1 + 10^((s_B − s_A)/400)), drawn as a Bernoulli
  trial. The `outcome_mode="expected"` setting removes the noise; it is used for
  validation.
- **Queue:** each tick, an idle player joins with probability 0.3. Players
  are paired with their closest-rated neighbour if the gap is within
  `50 + 25 × ticks waited` points (capped at 600).
- **Randomness:** one seed spawns separate streams for the population, queue
  entry, matchmaking and outcomes, so arms run on the same seed share a
  population and the comparison is paired.

## Design notes

- **Centred RMSE.** Elo identifies only skill differences, so a common offset
  is removed before computing error. Raw RMSE is reported as well.
- **Settling time, not first passage.** "Games to converge" is the game after
  which a player's error stays within ±100 for the rest of the run. Players
  still outside at the end are right-censored. Starting near the default
  rating does not count as having converged.
- **Matchmaking compresses the rating scale (found while building, see H3).**
  With noiseless outcomes and closest-rating pairing, Elo orders 10k players
  almost perfectly (Spearman ≈ 1.0), but the rating spread barely grows: RMSE
  stays around 230. Players only ever face neighbours, so information about
  scale spreads very slowly up the ladder. With random pairing the same Elo
  converges. The validation test therefore uses random pairing, and
  matchmaking policy is treated as an experimental factor.

## Limitations (so far)

- Matches are 1v1. Team games such as Valorant, CoD and PUBG squads aggregate
  skill in ways this model does not capture.
- True skill is fixed (no learning curve), and outcomes follow the same
  logistic curve Elo assumes, so Elo is correctly specified here. A
  misspecified outcome model (probit, or skill varying with map or role) is
  planned as a robustness check.
- Activity is uniform, with no peak hours or player-specific play rates.
