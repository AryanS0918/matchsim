"""Baseline: fixed-K Elo vs decaying-K Elo, replicated over seeds.

    python scripts/run_baseline.py --seeds 10 --players 10000 --ticks 500

Writes results/baseline_summaries.csv, results/baseline_history.csv and
results/baseline_config.json, then prints mean +/- 95% CI per arm and the
paired difference. All numbers are from simulated players.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from matchsim import Elo, EloDecayingK, SimConfig
from matchsim.replicate import paired_difference, run_replicates, summarise_replicates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--players", type=int, default=10_000)
    p.add_argument("--ticks", type=int, default=500)
    p.add_argument("--matchmaking", choices=["nearest", "random"], default="nearest")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    cfg = SimConfig(n_players=args.players, n_ticks=args.ticks, matchmaking=args.matchmaking)
    arms = {
        "elo_k32": lambda: Elo(k=32),
        "elo_decay": lambda: EloDecayingK(k_max=64, k_min=16, tau=30),
    }
    summaries, history = run_replicates(arms, cfg, seeds=range(args.seeds))

    args.out.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.out / "baseline_summaries.csv", index=False)
    history.to_csv(args.out / "baseline_history.csv", index=False)
    (args.out / "baseline_config.json").write_text(json.dumps(
        {"config": cfg.to_dict(), "arms": list(arms), "seeds": list(range(args.seeds))}, indent=2))

    pd.set_option("display.width", 120)
    print(f"\n{args.seeds} seeds x {len(arms)} arms, {args.players} players, "
          f"{args.ticks} ticks, matchmaking={args.matchmaking} (SIMULATED DATA)\n")
    print(summarise_replicates(summaries).round(3).to_string(index=False))
    print()
    diffs = [paired_difference(summaries, m, "elo_k32", "elo_decay")
             for m in ["final_rmse_centred", "mean_unfairness", "median_games_to_settle"]]
    print("Paired differences (same seed = same population):")
    print(pd.DataFrame(diffs).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
