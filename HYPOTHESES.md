# Pre-registered hypotheses

Written before the main experiments. Results get reported against these as
stated, including the ones that fail. Anything added after looking at
outputs is marked **exploratory**.

All results come from a simulation. They describe how the rating systems
behave under the assumptions in this model, not how real players behave.

## Design common to all hypotheses

- **Arms:** `elo_k32` (fixed K), `elo_decay` (K from 64 to 16, tau = 30),
  `glicko2`, and optionally `trueskill_lite`.
- **Unit of replication:** one full simulation run (one seed). Matches within
  a run are not independent, so match-level CIs would be far too narrow.
- **Paired design:** every arm runs on the same seeds, so each seed gives
  every arm the same population and the same queue-entry draws (common random
  numbers). Contrasts are paired differences across seeds.
- **Primary metrics:**
  - centred rating RMSE at a fixed horizon
  - games to settle within ±100 of true skill, as a per-player time to event,
    right-censored, summarised with a Kaplan–Meier median
  - mean match unfairness |P_true(A wins) − 0.5|
- **Smallest effect of interest (SESOI):** 5 RMSE points, 5 games to settle,
  0.01 unfairness. A difference that is statistically clear but smaller than
  this gets reported as "no practical difference".
- **Inference:** 95% t-intervals on paired differences, with Holm correction
  across the primary contrasts (H1, H2, H4, H5, H7).
- **Number of seeds:** set by a power calculation on a 10-seed pilot,
  n = ((z₀.₉₇₅ + z₀.₈) · SD_diff / SESOI)², with a minimum of 20 seeds.

## Validation checks (must pass before any hypothesis is tested)

- **V1.** With noiseless outcomes and random pairing, Elo recovers true skill:
  centred RMSE < 5 and Spearman > 0.999. *(Done in `tests/test_simulation.py`.)*
- **V2.** Glicko-2 matches the worked example in Glickman's paper to 3 d.p.
- **V3.** In a population where everyone has equal skill, unfairness is about 0
  and ratings only drift as a random walk around the default.
- **V4.** Smurf, tilt and churn switched off reproduce the baseline exactly
  (same seed, same history).

## Hypotheses

**H1. Convergence speed.** The Kaplan–Meier median games to settle are
ordered `glicko2` < `elo_decay` < `elo_k32`.
*Why:* Glicko-2 and decaying-K both make large early updates for new players.
Glicko-2 also scales each update to the opponent's uncertainty.
*Falsified if:* the `glicko2` vs `elo_decay` difference has a CI that includes 0
or is smaller than the SESOI.

**H2. The speed–accuracy trade-off in K.** For fixed-K Elo, a larger K settles
faster but leaves a higher steady-state RMSE floor, so there is a U-shaped
optimum in RMSE at any given horizon. `elo_decay` and `glicko2` lie on or
outside the fixed-K frontier, meaning they are at least as good on both axes.
*Test:* a sensitivity grid with K ∈ {8, 16, 24, 32, 48, 64}.

**H3 (exploratory). Matchmaking compresses the rating scale.** We found this
while building the code, so it is not confirmatory. Under nearest-rating
matchmaking, systems rank players well (high Spearman) but underestimate the
spread (rating SD / true SD < 1). This shows up as a large gap between RMSE
and rank error compared with random pairing.
*Prediction:* the compression is weakest for `glicko2`.

**H4. Fairness is decided early.** The between-system differences in
unfairness are concentrated in each player's first 20 games. After 100
games, the differences fall below the SESOI.
*Test:* unfairness by games-played bucket.

**H5. Smurfs.** A smurf (true skill +2 SD, starting at the default rating)
reaches its true rating band fastest under `glicko2`, then `elo_decay`, then
`elo_k32`. The harm to other players, measured as opponent-matches with
P_true > 0.75 per smurf, is roughly proportional to that time.

**H6. Tilt.** Tilt (a skill penalty after L consecutive losses) raises both RMSE
and unfairness in every system, and the effect grows with K, because a
high-K system chases temporary dips. Glicko-2 volatility rises for players who
tilt often.
*Test:* a sensitivity grid over tilt strength × K. The prediction is that the
effects are monotone in both.

**H7. Retention.** With churn driven by losing streaks, a Cox PH model on
time-to-churn gives hazard ratios below 1 for both `glicko2` and `elo_decay`
against `elo_k32`. The effect is largest for players whose true skill is far
from the default rating.
*Caveat to state alongside the result:* churn here is a mechanism we assume, so
this measures how much the choice of rating system matters *given* that
mechanism. The result is reported across several churn functions, not just one.

## A note on the A/B framing

In a real A/B test you cannot see true skill. The metrics a live experiment
could actually observe are prediction log-loss on match outcomes, the
favourite's win rate, queue time and retention. Each of those is also computed
here, so the simulation can say which observable metric would have detected the
true-skill differences, and with how many matches.
