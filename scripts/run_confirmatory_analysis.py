"""Run confirmatory 20-seed H1/H2 analysis and Glicko-2 tau sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from matchsim import Elo, EloDecayingK, Glicko2, SimConfig
from matchsim.replicate import mean_ci, paired_difference, run_replicates


FIXED_K = (8, 16, 24, 32, 48, 64)
TAUS = (0.3, 0.6, 1.2)
PRIMARY_METRICS = ("final_rmse_centred", "median_games_to_settle")


def holm_adjust(pvalues: list[float]) -> list[float]:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(pvalues.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted[index] = max(running, (pvalues.size - rank) * pvalues[index])
        running = adjusted[index]
    return np.minimum(adjusted, 1.0).tolist()


def paired_ttest(summaries: pd.DataFrame, metric: str, arm_a: str,
                 arm_b: str) -> dict:
    result = paired_difference(summaries, metric, arm_a, arm_b)
    wide = summaries.pivot(index="seed", columns="arm", values=metric)
    difference = wide[arm_a] - wide[arm_b]
    test = stats.ttest_1samp(difference, popmean=0)
    threshold = 5.0
    excludes_sesoi = (
        result["ci_low"] > threshold or result["ci_high"] < -threshold
    )
    return {
        **result,
        "p_value": float(test.pvalue),
        "sesoi": threshold,
        "ci_excludes_sesoi": bool(excludes_sesoi),
    }


def run_confirmatory(config: SimConfig, seeds: list[int], out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    arms = {f"elo_k{k}": lambda k=k: Elo(k=k) for k in FIXED_K}
    arms["elo_decay"] = lambda: EloDecayingK(k_max=64, k_min=16, tau=30)
    arms["glicko2_tau0.6"] = lambda: Glicko2(tau=0.6, name="glicko2_tau0.6")
    arms["glicko2_tau0.3"] = lambda: Glicko2(tau=0.3, name="glicko2_tau0.3")
    arms["glicko2_tau1.2"] = lambda: Glicko2(tau=1.2, name="glicko2_tau1.2")

    summaries, history = run_replicates(arms, config, seeds)
    rows = []
    contrasts = (
        ("glicko2_tau0.6", "elo_decay"),
        ("glicko2_tau0.6", "elo_k32"),
        ("elo_decay", "elo_k32"),
    )
    for arm_a, arm_b in contrasts:
        for metric in PRIMARY_METRICS:
            rows.append(paired_ttest(summaries, metric, arm_a, arm_b))
    contrasts_df = pd.DataFrame(rows)
    contrasts_df["holm_p_value"] = holm_adjust(contrasts_df["p_value"].tolist())
    contrasts_df["holm_reject"] = contrasts_df["holm_p_value"] < 0.05
    contrasts_df.to_csv(out / "confirmatory_holm_contrasts.csv", index=False)

    arm_rows = []
    for arm, group in summaries.groupby("arm", sort=False):
        row = {"arm": arm, "n_seeds": int(group["seed"].nunique())}
        for metric in PRIMARY_METRICS:
            mean, low, high = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        arm_rows.append(row)
    arm_df = pd.DataFrame(arm_rows)
    arm_df.to_csv(out / "confirmatory_arm_results.csv", index=False)
    return arm_df, contrasts_df


def tau_sweep(config: SimConfig, seeds: list[int], out: Path) -> pd.DataFrame:
    arms = {f"glicko2_tau{tau:g}": lambda tau=tau: Glicko2(
        tau=tau, name=f"glicko2_tau{tau:g}") for tau in TAUS}
    summaries, _ = run_replicates(arms, config, seeds)
    rows = []
    for arm, group in summaries.groupby("arm", sort=False):
        tau = float(arm.removeprefix("glicko2_tau"))
        row = {"arm": arm, "tau": tau, "n_seeds": int(group["seed"].nunique())}
        for metric in PRIMARY_METRICS:
            mean, low, high = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("tau")
    result.to_csv(out / "glicko2_tau_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, title, ylabel, axis in (
        ("final_rmse_centred", "Centred RMSE", "RMSE (lower is better)", axes[0]),
        ("median_games_to_settle", "Games to settle", "Median games (lower is faster)", axes[1]),
    ):
        means = result[f"{metric}_mean"]
        lows = result[f"{metric}_ci_low"]
        highs = result[f"{metric}_ci_high"]
        axis.errorbar(result["tau"], means, yerr=[means - lows, highs - means],
                      marker="o", capsize=4)
        axis.set_xticks(TAUS)
        axis.set(title=f"Glicko-2 tau sweep: {title}", xlabel="tau", ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(out / "glicko2_tau_sweep.png", dpi=160)
    plt.close(fig)
    return result


def plot_confirmatory(grid: pd.DataFrame, out: Path) -> None:
    fixed = grid[grid["arm"].str.startswith("elo_k")].copy()
    fixed["k"] = fixed["arm"].str.extract(r"elo_k(\d+)").astype(float)
    fixed = fixed.sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, title, ylabel, axis in (
        ("final_rmse_centred", "Centred RMSE", "RMSE (lower is better)", axes[0]),
        ("median_games_to_settle", "Games to settle", "Median games (lower is faster)", axes[1]),
    ):
        means = fixed[f"{metric}_mean"]
        lows = fixed[f"{metric}_ci_low"]
        highs = fixed[f"{metric}_ci_high"]
        axis.errorbar(fixed["k"], means, yerr=[means - lows, highs - means],
                      marker="o", capsize=4)
        axis.set_xticks(FIXED_K)
        axis.set(title=f"H2 confirmatory grid: {title}", xlabel="K", ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(out / "confirmatory_k_sensitivity.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.seeds < 20:
        parser.error("--seeds must be at least 20 for confirmatory analysis")

    config = SimConfig(n_players=args.players, n_ticks=args.ticks)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    arm_results, contrasts = run_confirmatory(config, list(range(args.seeds)), out)
    tau_results = tau_sweep(config, list(range(args.seeds)), out)
    plot_confirmatory(arm_results, out)

    (out / "confirmatory_config.json").write_text(json.dumps({
        "config": config.to_dict(),
        "seeds": list(range(args.seeds)),
        "fixed_k": list(FIXED_K),
        "taus": list(TAUS),
        "holm_family": "H1/H2 primary paired contrasts",
        "sesoi_rmse": 5.0,
        "sesoi_games": 5.0,
    }, indent=2) + "\n")

    print(arm_results.round(3).to_string(index=False))
    print("\nHolm-adjusted H1/H2 contrasts:")
    print(contrasts[[
        "metric", "contrast", "mean_diff", "ci_low", "ci_high",
        "p_value", "holm_p_value", "holm_reject", "ci_excludes_sesoi",
    ]].round(6).to_string(index=False))
    print("\nTau sweep:")
    print(tau_results[[
        "tau", "final_rmse_centred_mean", "median_games_to_settle_mean",
    ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
