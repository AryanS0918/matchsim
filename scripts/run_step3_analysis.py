"""Run the 10-seed K-sensitivity and Glicko-2 pilot.

    python scripts/run_step3_analysis.py --players 2000 --ticks 500 --seeds 10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matchsim import Elo, EloDecayingK, Glicko2, SimConfig
from matchsim.replicate import mean_ci, paired_difference, run_replicates


FIXED_K = (8, 16, 24, 32, 48, 64)
PILOT_METRICS = (
    "final_rmse_centred",
    "median_games_to_settle",
    "prediction_log_loss",
    "favorite_win_rate",
    "mean_queue_time",
)


def required_seeds(sd: float, effect: float, minimum: int = 1) -> tuple[float, int]:
    if not np.isfinite(sd) or not np.isfinite(effect):
        return float("nan"), minimum
    raw = ((1.96 + 0.84) * sd / abs(effect)) ** 2 if effect else float("inf")
    return raw, max(minimum, math.ceil(raw)) if np.isfinite(raw) else minimum


def aggregate(summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in summaries.groupby("arm", sort=False):
        row = {"arm": arm, "n_seeds": int(group["seed"].nunique())}
        for metric in PILOT_METRICS:
            mean, low, high = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm_a, arm_b in (("glicko2", "elo_decay"), ("glicko2", "elo_k32")):
        for metric in PILOT_METRICS:
            result = paired_difference(summaries, metric, arm_a, arm_b)
            differences = summaries.pivot(index="seed", columns="arm", values=metric)
            values = (differences[arm_a] - differences[arm_b]).to_numpy()
            sd = float(np.nanstd(values, ddof=1))
            threshold = {
                "final_rmse_centred": 5.0,
                "median_games_to_settle": 5.0,
            }.get(metric)
            raw_n, with_minimum = (
                required_seeds(sd, threshold, minimum=20)
                if threshold is not None
                else (float("nan"), float("nan"))
            )
            observable_raw_n, observable_n = (
                required_seeds(sd, result["mean_diff"], minimum=20)
                if metric in {"prediction_log_loss", "favorite_win_rate", "mean_queue_time"}
                else (float("nan"), float("nan"))
            )
            excludes_zero = (result["ci_low"] > 0) or (result["ci_high"] < 0)
            rows.append({
                **result,
                "sd_diff": sd,
                "ci_excludes_zero": bool(excludes_zero),
                "sesoi": threshold,
                "clears_sesoi": bool(
                    threshold is not None and (
                        result["ci_low"] > threshold
                        or result["ci_high"] < -threshold
                    )
                ),
                "power_n_raw": raw_n,
                "power_n_with_min_20": with_minimum,
                "observed_effect_power_n": observable_raw_n,
                "observed_effect_power_n_with_min_20": observable_n,
            })
    return pd.DataFrame(rows)


def plot_sensitivity(grid: pd.DataFrame, output: Path) -> None:
    decay = grid.loc[grid["arm"] == "elo_decay"].iloc[0]
    fixed = grid.loc[grid["k"].notna()].sort_values("k")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        fixed["k"], fixed["final_rmse_centred_mean"],
        yerr=[
            fixed["final_rmse_centred_mean"] - fixed["final_rmse_centred_ci_low"],
            fixed["final_rmse_centred_ci_high"] - fixed["final_rmse_centred_mean"],
        ],
        marker="o", capsize=3, label="fixed-K Elo",
    )
    ax.axhline(decay["final_rmse_centred_mean"], color="tab:orange", linestyle="--",
               label="elo_decay mean")
    ax.fill_between(
        ax.get_xlim(), decay["final_rmse_centred_ci_low"],
        decay["final_rmse_centred_ci_high"], color="tab:orange", alpha=0.15,
    )
    ax.set(title="Centred RMSE by fixed Elo K", xlabel="K factor",
           ylabel="Final centred RMSE (lower is better)")
    ax.set_xticks(FIXED_K)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_settling(grid: pd.DataFrame, output: Path) -> None:
    decay = grid.loc[grid["arm"] == "elo_decay"].iloc[0]
    fixed = grid.loc[grid["k"].notna()].sort_values("k")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        fixed["k"], fixed["median_games_to_settle_mean"],
        yerr=[
            fixed["median_games_to_settle_mean"] - fixed["median_games_to_settle_ci_low"],
            fixed["median_games_to_settle_ci_high"] - fixed["median_games_to_settle_mean"],
        ],
        marker="o", capsize=3, label="fixed-K Elo",
    )
    ax.axhline(decay["median_games_to_settle_mean"], color="tab:orange", linestyle="--",
               label="elo_decay mean")
    ax.fill_between(
        ax.get_xlim(), decay["median_games_to_settle_ci_low"],
        decay["median_games_to_settle_ci_high"], color="tab:orange", alpha=0.15,
    )
    ax.set(title="Kaplan–Meier median games to settle by Elo K",
           xlabel="K factor", ylabel="Median games to settle (lower is faster)")
    ax.set_xticks(FIXED_K)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2 for confidence intervals")

    config = SimConfig(n_players=args.players, n_ticks=args.ticks)
    arms = {f"elo_k{k}": lambda k=k: Elo(k=k) for k in FIXED_K}
    arms["elo_decay"] = lambda: EloDecayingK(k_max=64, k_min=16, tau=30)
    arms["glicko2"] = Glicko2
    seeds = list(range(args.seeds))

    summaries, _ = run_replicates(arms, config, seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    arm_results = aggregate(summaries)
    arm_results.to_csv(args.out / "step3_arm_results.csv", index=False)
    (args.out / "step3_pilot_config.json").write_text(json.dumps({
        "config": config.to_dict(),
        "arms": list(arms),
        "seeds": seeds,
        "settling_median": "per-seed Kaplan-Meier median with right censoring",
        "glicko2_rating_period": "one simulation tick",
    }, indent=2) + "\n")

    grid = arm_results.copy()
    grid["k"] = grid["arm"].str.extract(r"^elo_k(\d+)$").astype(float)
    grid.to_csv(args.out / "k_sensitivity.csv", index=False)
    plot_sensitivity(grid, args.out / "k_sensitivity_rmse.png")
    plot_settling(grid, args.out / "k_sensitivity_settling.png")

    paired = contrasts(summaries)
    paired.to_csv(args.out / "h1_paired_contrasts.csv", index=False)

    print(f"{args.seeds} paired seeds; {args.players} players; {args.ticks} ticks")
    print(arm_results[[
        "arm", "final_rmse_centred_mean", "final_rmse_centred_ci_low",
        "final_rmse_centred_ci_high", "median_games_to_settle_mean",
        "median_games_to_settle_ci_low", "median_games_to_settle_ci_high",
        "prediction_log_loss_mean", "favorite_win_rate_mean", "mean_queue_time_mean",
    ]].round(3).to_string(index=False))
    print("\nH1 contrasts and power:")
    print(paired[[
        "metric", "contrast", "mean_diff", "ci_low", "ci_high",
        "ci_excludes_zero", "sesoi", "clears_sesoi", "power_n_raw",
        "power_n_with_min_20", "observed_effect_power_n",
        "observed_effect_power_n_with_min_20",
    ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
