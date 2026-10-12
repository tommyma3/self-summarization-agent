"""Plot compact value MC training dynamics versus default GRPO.

Reads, per run:
  artifacts/train/<run>/eval_metrics.jsonl   — per-checkpoint eval metrics
  artifacts/train/<run>/step_metrics.jsonl   — per-training-step metrics
  artifacts/train/<run>/rollouts/iteration-NNNNN.judged.jsonl — judged train rollouts

Outputs a 3x3 comparison grid plus standalone SVGs for the headline curves.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter


STYLES = [
    {"label": "Default GRPO", "color": "#2878b5", "marker": "o", "linestyle": "-"},
    {"label": "Compact value MC", "color": "#b44c26", "marker": "s", "linestyle": "--"},
]

# Grid panels: (key, title, ylabel). "train_accuracy"/"train_malformed_rate" come from
# judged rollouts; everything else is a step_metrics.jsonl field.
PANELS = [
    ("eval_accuracy", "Eval accuracy per checkpoint (higher is better)", "Accuracy"),
    ("train_accuracy", "Train rollout accuracy per checkpoint (higher is better)", "Accuracy"),
    ("train_malformed_rate", "Train malformed-outcome rate (lower is better)", "Fraction of train rollouts"),
    ("mean_reward", "Mean training reward per step", "Reward"),
    ("value/loss", "Value loss per step (compact value MC)", "Loss"),
    ("value/classification_accuracy", "Value classification accuracy per step (compact value MC)", "Accuracy"),
    ("mean_policy_kl", "Policy KL per step", "KL"),
    ("clip_fraction", "PPO clip fraction per step", "Fraction"),
    ("loss", "Total training loss per step", "Loss"),
]

STANDALONE = [
    ("eval_accuracy", "eval_accuracy.svg"),
    ("train_accuracy", "train_accuracy.svg"),
    ("value/loss", "value_loss.svg"),
    ("value/classification_accuracy", "value_classification_accuracy.svg"),
]


def number(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value) if math.isfinite(value) else math.nan


def load_eval_metrics(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["iteration"])] = row
    if not rows:
        raise ValueError(f"No evaluations in {path}")
    return dict(sorted(rows.items()))


def load_step_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No steps in {path}")
    return rows


_OUTCOME_RE = re.compile(rb'"outcome":\s*"([^"]+)"')


def _outcomes(path: Path):
    """Yield judge outcomes; regex fast path, JSON fallback for ambiguous lines."""
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            matches = _OUTCOME_RE.findall(line)
            if len(matches) == 1:
                yield matches[0].decode("utf-8", "replace")
            else:
                yield json.loads(line)["judge"]["outcome"]


def load_train_rollout_stats(rollouts_dir: Path) -> dict[int, dict]:
    """Training rollout accuracy per sampled checkpoint.

    iteration-NNNNN.judged.jsonl holds rollouts sampled from checkpoint N-1,
    so the stats are keyed by checkpoint iteration (filename number minus one).
    """
    stats = {}
    for path in sorted(rollouts_dir.glob("iteration-*.judged.jsonl")):
        checkpoint = int(path.stem.split("-")[1].split(".")[0]) - 1
        total = correct = malformed = 0
        for outcome in _outcomes(path):
            total += 1
            if outcome == "correct_answer":
                correct += 1
            elif outcome == "malformed_tool_call":
                malformed += 1
        if total:
            stats[checkpoint] = {
                "accuracy": correct / total,
                "malformed": malformed,
                "total": total,
            }
    return dict(sorted(stats.items()))


def metric_series(step_rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for step, row in enumerate(step_rows, 1):
        value = number(row, key)
        if math.isfinite(value):
            xs.append(step)
            ys.append(value)
    return xs, ys


def plot_panel(ax, eval_runs, train_stats, step_runs, spec: tuple) -> None:
    key, title, ylabel = spec
    plotted = False
    if key == "eval_accuracy":
        for run, style in zip(eval_runs, STYLES):
            xs = list(run)
            ys = [run[i]["eval_accuracy"] for i in xs]
            ax.plot(xs, ys, **style, linewidth=1.8, markersize=3)
            plotted = True
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Evaluated checkpoint iteration")
    elif key in ("train_accuracy", "train_malformed_rate"):
        for stats, style in zip(train_stats, STYLES):
            xs = list(stats)
            if key == "train_accuracy":
                ys = [stats[i]["accuracy"] for i in xs]
            else:
                ys = [stats[i]["malformed"] / stats[i]["total"] for i in xs]
            if xs:
                ax.plot(xs, ys, **style, linewidth=1.8, markersize=3)
                plotted = True
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Sampled checkpoint iteration")
    else:
        all_ys = []
        for rows, style in zip(step_runs, STYLES):
            xs, ys = metric_series(rows, key)
            if xs:
                ax.plot(xs, ys, **style, linewidth=1.8, markersize=3)
                plotted = True
                all_ys.extend(ys)
        if all_ys and min(all_ys) < 0:
            ax.axhline(0, color="#666666", linewidth=1)
        else:
            ax.set_ylim(bottom=0)
        if "tokens" in key:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Training step")
    if not plotted:
        ax.text(0.5, 0.5, "Metric unavailable", transform=ax.transAxes, ha="center")
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def render_grid(eval_runs, train_stats, step_runs, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, spec in zip(axes.flat, PANELS):
        plot_panel(ax, eval_runs, train_stats, step_runs, spec)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Compact value MC vs default GRPO — training dynamics", fontsize=20, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=2, frameon=False)
    fig.text(0.5, 0.01, "Value-metric panels show the compact value MC run only; the GRPO run has no value model. "
             "Train rollout stats come from judged rollout files (gaps where a file is missing).",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=2, w_pad=2)
    for extension in ("png", "svg"):
        fig.savefig(output / f"training_dynamics.{extension}", dpi=160)
    plt.close(fig)


def render_standalone(eval_runs, train_stats, step_runs, output: Path) -> None:
    for key, filename in STANDALONE:
        fig, ax = plt.subplots(figsize=(10, 6))
        spec = next(spec for spec in PANELS if spec[0] == key)
        plot_panel(ax, eval_runs, train_stats, step_runs, spec)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)


def summary_text(eval_runs, train_stats, step_runs) -> str:
    lines = ["Compact value MC vs default GRPO — training dynamics", ""]
    mc_steps = step_runs[1]
    latest_value_loss = number(mc_steps[-1], "value/loss")
    latest_value_acc = number(mc_steps[-1], "value/classification_accuracy")
    lines.append(f"Value loss at final step: {latest_value_loss:.4f}" if math.isfinite(latest_value_loss)
                 else "Value loss at final step: unavailable")
    lines.append(f"Value classification accuracy at final step: {latest_value_acc:.2%}"
                 if math.isfinite(latest_value_acc)
                 else "Value classification accuracy at final step: unavailable")
    for stats, style in zip(train_stats, STYLES):
        if stats:
            last_iter = max(stats)
            lines.append(f"{style['label']}: final train rollout accuracy "
                         f"{stats[last_iter]['accuracy']:.1%} ({last_iter})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/train"))
    parser.add_argument("--default-run", default="qwen-bcplus-train", help="Run name or absolute run directory.")
    parser.add_argument("--mc-run", default="qwen-bcplus-compact-value-mc", help="Run name or absolute run directory.")
    parser.add_argument("--output-dir", type=Path, help="Default: <artifacts-dir>/compact_value_mc_training_curves")
    args = parser.parse_args()
    try:
        eval_runs = [load_eval_metrics(args.artifacts_dir / name / "eval_metrics.jsonl")
                     for name in (args.default_run, args.mc_run)]
        step_runs = [load_step_metrics(args.artifacts_dir / name / "step_metrics.jsonl")
                     for name in (args.default_run, args.mc_run)]
        train_stats = [load_train_rollout_stats(args.artifacts_dir / name / "rollouts")
                       for name in (args.default_run, args.mc_run)]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.artifacts_dir / "compact_value_mc_training_curves"
    output.mkdir(parents=True, exist_ok=True)
    render_grid(eval_runs, train_stats, step_runs, output)
    render_standalone(eval_runs, train_stats, step_runs, output)
    summary = summary_text(eval_runs, train_stats, step_runs)
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, file=sys.stderr)
    print(f"Plots and summary written to {output}")


if __name__ == "__main__":
    main()
