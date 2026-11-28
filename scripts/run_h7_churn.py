"""H7 churn simulation with loss-streak survival outcomes and Cox PH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from matchsim import Elo, EloDecayingK, Glicko2, SimConfig, run_simulation


CHURN_THRESHOLDS = (3, 5, 8)


def reconstruct_survival(res, threshold: int) -> pd.DataFrame:
    matches = res.matches
    if matches is None:
        raise RuntimeError("run H7 with record_matches=True")

    n = len(res.true_skill)
    streak = [0] * n
    churn_tick = [None] * n
    matches = matches.sort_values(["tick", "a", "b"])
    for row in matches.itertuples(index=False):
        score_a = float(row.score_a)
        a, b = int(row.a), int(row.b)
        if score_a >= 0.5:
            streak[b] += 1
            streak[a] = 0
        else:
            streak[a] += 1
            streak[b] = 0
        for player in (a, b):
            if churn_tick[player] is None and streak[player] >= threshold:
                churn_tick[player] = int(row.tick)

    horizon = int(res.config.n_ticks)
    rows = []
    for player in range(n):
        event_tick = churn_tick[player]
        rows.append({
            "seed": res.config.seed,
            "player": player,
            "arm": res.system_name,
            "duration": horizon if event_tick is None else event_tick,
            "event": int(event_tick is not None),
            "true_skill_distance": abs(float(res.true_skill[player] - res.config.skill_mean)),
            "elo_decay": int(res.system_name.startswith("elo_decay")),
            "glicko2": int(res.system_name.startswith("glicko2")),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")

    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise SystemExit("Install analysis dependencies with pip install -e '.[dev]'") from exc

    config = SimConfig(n_players=args.players, n_ticks=args.ticks)
    arms = {
        "elo_k32": lambda: Elo(k=32),
        "elo_decay": lambda: EloDecayingK(k_max=64, k_min=16, tau=30),
        "glicko2": lambda: Glicko2(tau=0.6),
    }
    all_rows = []
    for seed in range(args.seeds):
        cfg = config.with_(seed=seed)
        for arm_name, make_system in arms.items():
            result = run_simulation(make_system(), cfg, record_matches=True)
            all_rows.append((arm_name, result))

    output = []
    for threshold in CHURN_THRESHOLDS:
        frames = []
        for arm_name, result in all_rows:
            frame = reconstruct_survival(result, threshold)
            frame["arm"] = arm_name
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)

        cph = CoxPHFitter()
        cph.fit(
            data[[
                "duration", "event", "seed", "elo_decay", "glicko2",
                "true_skill_distance",
            ]],
            duration_col="duration",
            event_col="event",
            cluster_col="seed",
            robust=True,
        )
        summary = cph.summary
        for covariate in ("elo_decay", "glicko2"):
            row = summary.loc[covariate]
            output.append({
                "threshold": threshold,
                "comparator": "elo_k32",
                "arm": covariate,
                "hazard_ratio": float(row["exp(coef)"]),
                "ci_low": float(row["exp(coef) lower 95%"]),
                "ci_high": float(row["exp(coef) upper 95%"]),
                "p_value": float(row["p"]),
                "events": int(data["event"].sum()),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    hr = pd.DataFrame(output)
    hr.to_csv(args.out / "h7_cox_hazard_ratios.csv", index=False)
    (args.out / "h7_config.json").write_text(json.dumps({
        "config": config.to_dict(),
        "seeds": list(range(args.seeds)),
        "arms": list(arms),
        "churn_thresholds_consecutive_losses": list(CHURN_THRESHOLDS),
        "reference_arm": "elo_k32",
        "clustered_by_seed": True,
    }, indent=2) + "\n")
    print(hr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
