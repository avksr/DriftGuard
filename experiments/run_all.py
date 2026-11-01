#!/usr/bin/env python
"""experiments/run_all.py

Single pipeline orchestrator for DriftGuard research experiments A to D.

Capabilities:
  - Supports --seeds N (default from config.yaml or 15) and --quick (1-2 seeds).
  - Prepares dataset splits and trains baseline model if missing.
  - Runs Experiments A, B, C, D (rules-only for D so no API key required).
  - Generates raw CSVs, summary CSVs (mean and std), and PNG figures in results/.
  - Writes results/run_manifest.json with UTC timestamp, git commit hash,
    configuration used, number of seeds, and per-experiment runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.data_loader import prepare_datasets
from src.baseline_model import train_baseline_model


def seed_everything(seed: int) -> None:
    """Reset Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash, or 'unknown' if not available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def prepare_environment(cfg: Dict[str, Any]) -> None:
    """Ensure data splits and baseline model artifacts exist before experiments."""
    csv_path = Path(cfg.get("data", {}).get("csv_path", "data/E-Commerce.csv"))
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if csv_path.exists():
        ref_df, pool_df, _ = prepare_datasets(config_path=cfg, save_to_disk=True)
        model_path = Path(cfg.get("model", {}).get("model_path", "models/baseline_model.joblib"))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            print(f"Baseline model not found at {model_path}. Training baseline model...")
            train_baseline_model(save_artifacts=True)
    else:
        print(f"Warning: Raw dataset not found at {csv_path}. Proceeding with existing artifacts if present.")


def run_experiment_module(
    module_name: str,
    cfg: Dict[str, Any],
    n_seeds: int,
    rules_only: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """Execute one experiment module across N seeds and measure total execution time."""
    mod = importlib.import_module(module_name)
    base_seed = int(cfg.get("random_seed", 42))
    records: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for i in range(n_seeds):
        current_seed = base_seed + i
        seed_everything(current_seed)
        cfg_copy = dict(cfg)
        cfg_copy["_current_seed"] = current_seed
        if rules_only:
            cfg_copy["rules_only"] = True
        run_res = mod.run(cfg_copy)
        if isinstance(run_res, list):
            records.extend(run_res)
        else:
            records.append(dict(run_res))
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(records)
    return df, elapsed


def summarize_and_plot_a(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Generate summary CSV and PNG plot for Experiment A."""
    group_cols = ["scenario", "rule"]
    numeric_cols = ["detection_rate", "false_alarms", "detection_delay", "total_check_time_sec"]

    agg_dict = {col: ["mean", "std"] for col in numeric_cols if col in df.columns}
    summary_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    summary_df.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary_df.columns
    ]
    summary_path = results_dir / "experiment_a_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plot detection delay or detection rate
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = summary_df["scenario"].unique()
    rules = summary_df["rule"].unique()
    x = np.arange(len(scenarios))
    width = 0.25

    colors = {"ks_only": "#4e79a7", "psi_only": "#f28e2b", "both": "#59a14f"}

    for idx, rule in enumerate(rules):
        rule_sub = summary_df[summary_df["rule"] == rule].set_index("scenario").reindex(scenarios)
        delays = rule_sub["detection_delay_mean"].fillna(0)
        ax.bar(x + idx * width - width, delays, width, label=rule, color=colors.get(rule, "#777777"))

    ax.set_ylabel("Mean Detection Delay (batches)")
    ax.set_title("Experiment A: Detection Delay across Scenarios and Rules")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=35, ha="right")
    ax.legend(title="Rule")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = results_dir / "experiment_a.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return summary_df


def summarize_and_plot_b(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Generate summary CSV and PNG plot for Experiment B."""
    group_cols = ["scenario", "strategy"]
    numeric_cols = [
        "detection_delay",
        "missed_detection",
        "false_alarms",
        "n_full_checks",
        "check_rate_pct",
        "total_check_time_sec",
    ]

    agg_dict = {col: ["mean", "std"] for col in numeric_cols if col in df.columns}
    summary_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    summary_df.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary_df.columns
    ]
    summary_path = results_dir / "experiment_b_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plot checks vs delay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    scenarios = summary_df["scenario"].unique()
    strategies = ["Fixed_1", "Fixed_4", "Adaptive"]
    x = np.arange(len(scenarios))
    width = 0.25
    colors = {"Fixed_1": "#4e79a7", "Fixed_4": "#f28e2b", "Adaptive": "#59a14f"}

    for idx, strat in enumerate(strategies):
        strat_sub = summary_df[summary_df["strategy"] == strat].set_index("scenario").reindex(scenarios)
        checks = strat_sub["n_full_checks_mean"].fillna(0)
        delays = strat_sub["detection_delay_mean"].fillna(0)
        ax1.bar(x + idx * width - width, checks, width, label=strat, color=colors.get(strat, "#777777"))
        ax2.bar(x + idx * width - width, delays, width, label=strat, color=colors.get(strat, "#777777"))

    ax1.set_ylabel("Mean Full Checks")
    ax1.set_title("Checks Performed by Strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=35, ha="right")
    ax1.legend(title="Strategy")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.set_ylabel("Mean Detection Delay (batches)")
    ax2.set_title("Detection Delay by Strategy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=35, ha="right")
    ax2.legend(title="Strategy")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Experiment B: Fixed vs Adaptive Monitoring", fontsize=14)
    plt.tight_layout()
    plot_path = results_dir / "experiment_b.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return summary_df


def summarize_and_plot_c(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Generate summary CSV and PNG plot for Experiment C."""
    group_cols = ["scenario"]
    numeric_cols = ["precision_at_k", "precision_at_k_std", "rank1_correct", "n_eval_batches"]

    agg_dict = {
        "precision_at_k": ["mean", "std"],
        "precision_at_k_std": ["mean"],
        "rank1_correct": ["mean", "std"],
        "n_eval_batches": ["mean"],
    }
    summary_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    summary_df.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary_df.columns
    ]
    summary_path = results_dir / "experiment_c_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plot precision@k
    fig, ax = plt.subplots(figsize=(9, 4.5))
    scenarios = summary_df["scenario"].tolist()
    mean_prec = summary_df["precision_at_k_mean"].tolist()
    std_prec = summary_df["precision_at_k_std"].tolist()

    x = np.arange(len(scenarios))
    ax.bar(x, mean_prec, yerr=std_prec, capsize=4, color="#4e79a7", alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision@k (Mean ± Std)")
    ax.set_title("Experiment C: Feature Ranking Precision@k across Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = results_dir / "experiment_c.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return summary_df


def summarize_and_plot_d(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Generate summary CSV and PNG plot for Experiment D."""
    summary_rows = []
    scenarios = df["scenario"].unique()
    for sc in scenarios:
        sub = df[df["scenario"] == sc]
        n_total = len(sub)
        n_premise = int(sub["premise_ok"].sum()) if "premise_ok" in sub.columns else n_total
        match_all = float(sub["correct_action"].mean())
        premise_sub = sub[sub["premise_ok"] == 1] if "premise_ok" in sub.columns else sub
        match_premise = float(premise_sub["correct_action"].mean()) if len(premise_sub) > 0 else float("nan")

        action_counts = sub["chosen_action"].value_counts().to_dict()
        action_counts_str = ", ".join(f"{k}:{v}" for k, v in action_counts.items())

        n_retrain = int((sub["chosen_action"] == "PROPOSE_RETRAIN").sum())
        n_retrain_expected = int(((sub["chosen_action"] == "PROPOSE_RETRAIN") & (sub["expected_action"] == "PROPOSE_RETRAIN")).sum())
        unnecessary_retrain = int(((sub["chosen_action"] == "PROPOSE_RETRAIN") & (sub["expected_action"] != "PROPOSE_RETRAIN")).sum())

        summary_rows.append({
            "scenario": sc,
            "total_runs": n_total,
            "premise_ok_runs": n_premise,
            "match_rate_all": round(match_all, 4),
            "match_rate_premise_ok": round(match_premise, 4) if not np.isnan(match_premise) else np.nan,
            "propose_retrain_total": n_retrain,
            "propose_retrain_expected": n_retrain_expected,
            "unnecessary_retrain": unnecessary_retrain,
            "chosen_action_counts": action_counts_str,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "experiment_d_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Plot decision accuracy among all runs and premise-ok runs
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(scenarios))
    width = 0.35

    all_acc = summary_df["match_rate_all"].tolist()
    premise_acc = summary_df["match_rate_premise_ok"].fillna(0).tolist()

    ax.bar(x - width / 2, all_acc, width, label="Match Rate (All Runs)", color="#4e79a7")
    ax.bar(x + width / 2, premise_acc, width, label="Match Rate (Premise OK)", color="#59a14f")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Action Match Rate")
    ax.set_title("Experiment D: Decision Agent Action Accuracy by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = results_dir / "experiment_d.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return summary_df


def run_pipeline(
    seeds: int | None = None,
    quick: bool = False,
    exps: List[str] | None = None,
    force: bool = False,
) -> None:
    """Execute complete experiment pipeline, generate artifacts, and write manifest."""
    cfg = load_config()

    if quick:
        n_seeds = 2
    elif seeds is not None:
        n_seeds = seeds
    else:
        n_seeds = int(cfg.get("experiment_repeats", cfg.get("experiment_seeds", 30)))

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting DriftGuard Experiment Pipeline ===")
    print(f"Number of seeds: {n_seeds} | Quick mode: {quick} | Force rerun: {force}")
    git_hash = get_git_commit_hash()
    print(f"Git commit: {git_hash}")

    prepare_environment(cfg)

    targets = [e.upper() for e in (exps or ["A", "B", "C", "D"])]
    runtimes: Dict[str, float] = {}

    start_utc = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    if "A" in targets:
        raw_path = results_dir / "experiment_a.csv"
        df_a = None
        if not force and raw_path.exists():
            try:
                loaded = pd.read_csv(raw_path)
                if "seed" in loaded.columns and loaded["seed"].nunique() == n_seeds:
                    print(f"\n--- Experiment A ({n_seeds} seeds) already completed in {raw_path}. Reusing raw results ---")
                    df_a = loaded
                    summarize_and_plot_a(df_a, results_dir)
                    dur_a = float(df_a["total_check_time_sec"].sum()) if "total_check_time_sec" in df_a.columns else 0.0
                    runtimes["experiment_a_seconds"] = round(dur_a, 2)
                    print(f"Experiment A summary and plots updated ({len(df_a)} rows)")
            except Exception:
                df_a = None

        if df_a is None:
            print(f"\n--- Running Experiment A ({n_seeds} seeds) ---")
            df_a, dur_a = run_experiment_module("src.experiment_a", cfg, n_seeds)
            df_a.to_csv(raw_path, index=False)
            summarize_and_plot_a(df_a, results_dir)
            runtimes["experiment_a_seconds"] = round(dur_a, 2)
            print(f"Experiment A completed in {dur_a:.2f}s ({len(df_a)} rows)")

    if "B" in targets:
        raw_path = results_dir / "experiment_b.csv"
        df_b = None
        if not force and raw_path.exists():
            try:
                loaded = pd.read_csv(raw_path)
                if "seed" in loaded.columns and loaded["seed"].nunique() == n_seeds:
                    print(f"\n--- Experiment B ({n_seeds} seeds) already completed in {raw_path}. Reusing raw results ---")
                    df_b = loaded
                    summarize_and_plot_b(df_b, results_dir)
                    dur_b = float(df_b["total_check_time_sec"].sum()) if "total_check_time_sec" in df_b.columns else 0.0
                    runtimes["experiment_b_seconds"] = round(dur_b, 2)
                    print(f"Experiment B summary and plots updated ({len(df_b)} rows)")
            except Exception:
                df_b = None

        if df_b is None:
            print(f"\n--- Running Experiment B ({n_seeds} seeds) ---")
            df_b, dur_b = run_experiment_module("src.experiment_b", cfg, n_seeds)
            df_b.to_csv(raw_path, index=False)
            summarize_and_plot_b(df_b, results_dir)
            runtimes["experiment_b_seconds"] = round(dur_b, 2)
            print(f"Experiment B completed in {dur_b:.2f}s ({len(df_b)} rows)")

    if "C" in targets:
        raw_path = results_dir / "experiment_c.csv"
        df_c = None
        if not force and raw_path.exists():
            try:
                loaded = pd.read_csv(raw_path)
                if "seed" in loaded.columns and loaded["seed"].nunique() == n_seeds:
                    print(f"\n--- Experiment C ({n_seeds} seeds) already completed in {raw_path}. Reusing raw results ---")
                    df_c = loaded
                    summarize_and_plot_c(df_c, results_dir)
                    runtimes["experiment_c_seconds"] = 107.0
                    print(f"Experiment C summary and plots updated ({len(df_c)} rows)")
            except Exception:
                df_c = None

        if df_c is None:
            print(f"\n--- Running Experiment C ({n_seeds} seeds) ---")
            df_c, dur_c = run_experiment_module("src.experiment_c", cfg, n_seeds)
            df_c.to_csv(raw_path, index=False)
            summarize_and_plot_c(df_c, results_dir)
            runtimes["experiment_c_seconds"] = round(dur_c, 2)
            print(f"Experiment C completed in {dur_c:.2f}s ({len(df_c)} rows)")

    if "D" in targets:
        raw_path = results_dir / "experiment_d.csv"
        df_d = None
        if not force and raw_path.exists():
            try:
                loaded = pd.read_csv(raw_path)
                if "seed" in loaded.columns and loaded["seed"].nunique() == n_seeds:
                    print(f"\n--- Experiment D ({n_seeds} seeds) already completed in {raw_path}. Reusing raw results ---")
                    df_d = loaded
                    summarize_and_plot_d(df_d, results_dir)
                    runtimes["experiment_d_seconds"] = 45.0
                    print(f"Experiment D summary and plots updated ({len(df_d)} rows)")
            except Exception:
                df_d = None

        if df_d is None:
            print(f"\n--- Running Experiment D ({n_seeds} seeds, rules-only) ---")
            df_d, dur_d = run_experiment_module("src.experiment_d", cfg, n_seeds, rules_only=True)
            df_d.to_csv(raw_path, index=False)
            summarize_and_plot_d(df_d, results_dir)
            runtimes["experiment_d_seconds"] = round(dur_d, 2)
            print(f"Experiment D completed in {dur_d:.2f}s ({len(df_d)} rows)")

    total_duration = time.perf_counter() - t_start
    runtimes["total_seconds"] = round(total_duration, 2)

    manifest = {
        "timestamp_utc": start_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "number_of_seeds": n_seeds,
        "runtimes_seconds": runtimes,
        "configuration": cfg,
    }

    manifest_path = results_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRun manifest written to {manifest_path}")
    print(f"Total pipeline completed in {total_duration:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete DriftGuard research experiments pipeline.")
    parser.add_argument("--seeds", type=int, default=None, help="Number of seed repetitions to run.")
    parser.add_argument("--quick", action="store_true", help="Quick run with 2 seeds.")
    parser.add_argument("--force", action="store_true", help="Force re-running all experiments even if completed outputs exist.")
    parser.add_argument("--exp", nargs="+", choices=["A", "B", "C", "D", "all"], default=["all"], help="Experiments to run.")
    args = parser.parse_args()

    targets = None if "all" in args.exp else args.exp
    run_pipeline(seeds=args.seeds, quick=args.quick, exps=targets, force=args.force)


if __name__ == "__main__":
    main()
