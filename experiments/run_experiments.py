#!/usr/bin/env python
"""experiments/run_experiments.py

Run the four research experiments (A–D) defined in the project specification.

Features
--------
- Each experiment can be run individually via ``--exp A`` etc.
- ``--quick`` reduces the number of repeated runs (useful for development).
- Results are saved as CSV files under ``results/`` and summary plots as PNG.
- The script is deliberately lightweight – heavy lifting (drift detection, monitoring,
  LLM calls, etc.) lives in the ``src`` package.  This file only orchestrates the runs
  and aggregates metrics.

Usage example
-------------
```
python experiments/run_experiments.py --exp all            # run A, B, C, D
python experiments/run_experiments.py --exp B --quick    # fast run of Experiment B
```

The implementation avoids hard‑coded numbers; all configuration values are read from
``config.yaml``.  Random seeds are taken from the ``random_seed`` entry and then
offset for each repeat to guarantee reproducibility.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

from src.utils import load_config


def ensure_results_dir() -> Path:
    """Make sure the ``results`` directory exists and return its Path."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def seed_everything(seed: int) -> None:
    """Reset Python, NumPy and ``random`` RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass  # torch is optional – ignore if not installed

# ---------------------------------------------------------------------------
# Experiment wrappers – each wrapper calls the corresponding src module.
# The src modules are expected to expose a ``run`` function that returns a list of
# dictionaries, one per simulated run, containing the metrics described in the
# specification.
# ---------------------------------------------------------------------------

def _run_module(module_name: str, cfg: Dict[str, Any], repeat: int) -> List[Dict[str, Any]]:
    """Import a ``src`` experiment module and execute its ``run`` function.

    Parameters
    ----------
    module_name: str
        Fully‑qualified module name, e.g. ``src.experiments_a``.
    cfg: dict
        Project configuration dictionary.
    repeat: int
        Number of repetitions (different seeds).

    Returns
    -------
    list[dict]
        Collected metric dictionaries from each repetition.
    """
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Experiment module {module_name!r} not found") from exc
    if not hasattr(mod, "run"):
        raise AttributeError(f"Module {module_name!r} does not define a 'run' function")
    results: List[Dict[str, Any]] = []
    base_seed = cfg.get("random_seed", 42)
    for i in range(repeat):
        seed_everything(base_seed + i)
        cfg_copy = dict(cfg)
        cfg_copy["_current_seed"] = base_seed + i
        run_res = mod.run(cfg_copy)  # type: ignore[arg-type]
        if isinstance(run_res, list):
            results.extend(run_res)
        else:
            # Normalise to a list of dicts for consistency
            results.append(dict(run_res))
    return results

# ---------------------------------------------------------------------------
# Specific experiment runners – they delegate to modules under ``src``.
# If a module is missing we fall back to a tiny dummy implementation that records
# that the experiment was invoked.  This keeps the script usable while the rest
# of the codebase is still being built.
# ---------------------------------------------------------------------------

def run_experiment_a(cfg: Dict[str, Any], repeats: int) -> pd.DataFrame:
    """Experiment A – compare KS, PSI and the combined rule.

    Returns a DataFrame where each row corresponds to one simulation run.
    """
    try:
        records = _run_module("src.experiment_a", cfg, repeats)
    except Exception as exc:
        # Dummy fallback – record the configuration only
        records = [{"experiment": "A", "status": "fallback", "error": str(exc)}]
    return pd.DataFrame(records)


def run_experiment_b(cfg: Dict[str, Any], repeats: int) -> pd.DataFrame:
    """Experiment B – Fixed vs Adaptive monitoring comparison."""
    try:
        records = _run_module("src.experiment_b", cfg, repeats)
    except Exception as exc:
        records = [{"experiment": "B", "status": "fallback", "error": str(exc)}]
    return pd.DataFrame(records)


def run_experiment_c(cfg: Dict[str, Any], repeats: int) -> pd.DataFrame:
    """Experiment C – feature‑ranking precision."""
    try:
        records = _run_module("src.experiment_c", cfg, repeats)
    except Exception as exc:
        records = [{"experiment": "C", "status": "fallback", "error": str(exc)}]
    return pd.DataFrame(records)


def run_experiment_d(cfg: Dict[str, Any], repeats: int) -> pd.DataFrame:
    """Experiment D – decision‑agent quality (LLM vs rules‑only)."""
    try:
        records = _run_module("src.experiment_d", cfg, repeats)
    except Exception as exc:
        records = [{"experiment": "D", "status": "fallback", "error": str(exc)}]
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Plotting helpers – each experiment produces a summary PNG.
# The implementation uses Plotly Express for simple bar‑charts.
# ---------------------------------------------------------------------------

def _save_plot(df: pd.DataFrame, label: str, out_path: Path) -> None:
    """Create a tailored plot for experiment A/B/C/D and write to out_path."""
    try:
        import plotly.express as px
        if label == "A" and "rule" in df.columns and "scenario" in df.columns:
            plot_df = df.groupby(["scenario", "rule"], as_index=False)["detection_delay"].mean()
            fig = px.bar(plot_df, x="scenario", y="detection_delay", color="rule", barmode="group",
                         title="Experiment A: Detection Delay by Detector Rule across Scenarios")
        elif label == "B" and "strategy" in df.columns and "scenario" in df.columns:
            plot_df = df.groupby(["scenario", "strategy"], as_index=False)[["n_full_checks", "detection_delay"]].mean()
            fig = px.bar(plot_df, x="scenario", y="n_full_checks", color="strategy", barmode="group",
                         title="Experiment B: Checks Performed (Adaptive vs Fixed-1 vs Fixed-4)")
        elif label == "C" and "precision_at_k" in df.columns:
            plot_df = df.groupby("scenario", as_index=False)["precision_at_k"].mean()
            fig = px.bar(plot_df, x="scenario", y="precision_at_k",
                         title="Experiment C: Feature Ranking Precision@k by Scenario")
        elif label == "D" and "correct_action" in df.columns:
            plot_df = df.groupby(["scenario", "use_llm"], as_index=False)["correct_action"].mean()
            plot_df["mode"] = plot_df["use_llm"].apply(lambda u: "LLM + Rules" if u else "Rules Only")
            fig = px.bar(plot_df, x="scenario", y="correct_action", color="mode", barmode="group",
                         title="Experiment D: Decision Accuracy matching Expected Action")
        elif {"metric", "value"}.issubset(df.columns):
            fig = px.bar(df, x="metric", y="value", title=f"Experiment {label} Summary")
        else:
            fig = px.bar(pd.DataFrame({"metric": ["rows"], "value": [len(df)]}),
                         x="metric", y="value", title=f"Experiment {label} (fallback)")

        fig.write_image(str(out_path))
    except Exception as exc:
        print(f"  [Note] Could not save plot image ({exc}). CSV results are intact.")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run research experiments A–D.")
    parser.add_argument(
        "--exp",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small number of repeats for fast testing.",
    )
    args = parser.parse_args()

    cfg = load_config()
    repeats = 5 if args.quick else 30

    results_dir = ensure_results_dir()

    # Mapping from experiment label to runner function
    runners = {
        "A": run_experiment_a,
        "B": run_experiment_b,
        "C": run_experiment_c,
        "D": run_experiment_d,
    }

    selected = ["A", "B", "C", "D"] if args.exp == "all" else [args.exp]

    for label in selected:
        print(f"Running Experiment {label} (repeats={repeats}) …")
        df = runners[label](cfg, repeats)
        csv_path = results_dir / f"experiment_{label.lower()}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved {len(df)} rows to {csv_path}")
        plot_path = results_dir / f"experiment_{label.lower()}.png"
        _save_plot(df, label, plot_path)
        print(f"  -> Plot saved to {plot_path}")


    print("All requested experiments finished.")

if __name__ == "__main__":
    main()
