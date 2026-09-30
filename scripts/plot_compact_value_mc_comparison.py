"""Plot compact value MC versus default GRPO from read-only eval logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter


# Key, title, axis label; token/count averages are per evaluation rollout.
METRICS = [
    ("eval_accuracy", "Eval accuracy (higher is better)", "Accuracy"),
    ("accuracy_delta", "Accuracy difference at shared checkpoints", "MC − default (percentage points)"),
    ("malformed_rate", "Malformed-output rate (lower is better)", "Fraction of eval rollouts"),
    ("eval_avg_budget_consumed_tokens", "Budget tokens: generated + tool results", "Tokens / rollout"),
    ("eval_correct_per_1k_budget_consumed_tokens", "Token efficiency (higher is better)", "Correct / 1,000 budget tokens"),
    ("eval_avg_total_generated_tokens", "Total model-generated tokens", "Tokens / rollout"),
    ("eval_avg_summary_count", "Compaction frequency", "Summaries / rollout"),
    ("eval_avg_summary_generated_tokens", "Summary generation, including thinking", "Tokens / rollout"),
    ("eval_avg_search_calls", "Search usage", "Calls / rollout"),
]
STYLES = [
    {"label": "Default GRPO", "color": "#2878b5", "marker": "o", "linestyle": "-"},
    {"label": "Compact value MC", "color": "#b44c26", "marker": "s", "linestyle": "--"},
]


def number(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value) if math.isfinite(value) else math.nan


def metric_value(row: dict, key: str) -> float:
    if key == "malformed_rate":
        total = number(row, "eval_total")
        return number(row, "eval_malformed") / total if total > 0 else math.nan
    if key == "eval_correct_per_1k_budget_consumed_tokens" and key not in row:
        budget = number(row, "eval_avg_budget_consumed_tokens")
        return 1000 * number(row, "eval_accuracy") / budget if budget > 0 else math.nan
    return number(row, key)


def load_metrics(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                iteration = row["iteration"]
                if type(iteration) is not int or iteration < 0:
                    raise ValueError("iteration must be a nonnegative integer")
                if iteration in rows:
                    raise ValueError(f"duplicate iteration {iteration}; select one evaluation per checkpoint")
                if not 0 <= number(row, "eval_accuracy") <= 1:
                    raise ValueError("eval_accuracy must be a finite fraction in [0, 1]")
                if not number(row, "eval_total") > 0:
                    raise ValueError("eval_total must be positive")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            rows[iteration] = row
    if not rows:
        raise ValueError(f"No evaluations in {path}")
    return dict(sorted(rows.items()))


def comparison_notes(runs: list[dict], shared: list[int]) -> list[str]:
    notes = []
    if set(runs[0]) != set(runs[1]):
        notes.append("Checkpoint coverage differs; differences and summary use shared iterations only.")
    for key in ("eval_total", "eval_sampling_profile_id", "eval_samples_per_task"):
        values = [row.get(key) for run in runs for row in run.values()]
        if any(value is None for value in values):
            notes.append(f"Missing {key}; evaluation comparability cannot be fully checked.")
        elif len(set(values)) > 1:
            notes.append(f"{key} differs across evaluations; comparisons may not be like-for-like.")
    if 0 in shared and runs[0][0]["eval_accuracy"] != runs[1][0]["eval_accuracy"]:
        notes.append("Iteration-0 accuracies differ; the runs have different baseline behavior.")
    for run, style in zip(runs, STYLES):
        missing = [key for key, _, _ in METRICS if key != "accuracy_delta"
                   and any(not math.isfinite(metric_value(row, key)) for row in run.values())]
        if missing:
            notes.append(f"{style['label']}: missing metrics shown as gaps: {', '.join(missing)}.")
    return notes


def plot_panel(ax, runs: list[dict], shared: list[int], spec: tuple) -> None:
    key, title, ylabel = spec
    if key == "accuracy_delta":
        values = [100 * (runs[1][i]["eval_accuracy"] - runs[0][i]["eval_accuracy"]) for i in shared]
        ax.axhline(0, color="#666666", linewidth=1)
        ax.plot(shared, values, color=STYLES[1]["color"], marker="s", markersize=3)
        ax.fill_between(shared, values, 0, alpha=0.12, color=STYLES[1]["color"])
        ax.text(0.02, 0.96, "Above zero favors MC", transform=ax.transAxes, va="top", fontsize=9)
    else:
        for run, style in zip(runs, STYLES):
            values = [metric_value(row, key) for row in run.values()]
            ax.plot(list(run), values, **style, linewidth=1.8, markersize=3)
        ax.set_ylim(bottom=0)
        if key in ("eval_accuracy", "malformed_rate"):
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        elif "tokens" in key and "per_1k" not in key:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        if not any(math.isfinite(metric_value(row, key)) for run in runs for row in run.values()):
            ax.text(0.5, 0.5, "Metric unavailable", transform=ax.transAxes, ha="center")
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("Evaluated checkpoint iteration")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def render_plots(runs: list[dict], shared: list[int], output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, spec in zip(axes.flat, METRICS):
        plot_panel(ax, runs, shared, spec)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Compact value MC vs default GRPO", fontsize=20, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=2, frameon=False)
    fig.text(0.5, 0.01, "Raw evaluations; no smoothing. Differences use shared checkpoints. "
             "See summary.txt for comparison limits.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=2, w_pad=2)
    for extension in ("png", "svg"):
        fig.savefig(output / f"comparison.{extension}", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_panel(ax, runs, shared, METRICS[0])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "accuracy.svg")
    plt.close(fig)


def write_comparison(runs: list[dict], shared: list[int], output: Path) -> None:
    keys = [key for key, _, _ in METRICS if key != "accuracy_delta"]
    with (output / "matched_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration"] + [f"{key}_{suffix}" for key in keys
                                         for suffix in ("default", "mc", "mc_minus_default")])
        for iteration in shared:
            values = []
            for key in keys:
                default, mc = [metric_value(run[iteration], key) for run in runs]
                values.extend(value if math.isfinite(value) else "" for value in (default, mc, mc - default))
            writer.writerow([iteration, *values])


def summary_text(runs: list[dict], shared: list[int], window: int, notes: list[str]) -> str:
    tail = shared[-window:]
    lines = ["Compact value MC vs default GRPO", "",
             f"Shared checkpoints: {len(shared)} ({shared[0]}–{shared[-1]}).",
             f"Latest shared checkpoint: {shared[-1]}."]
    for run, style in zip(runs, STYLES):
        latest = run[shared[-1]]
        best = max(shared, key=lambda i: run[i]["eval_accuracy"])
        lines.append(
            f"{style['label']}: latest {latest['eval_accuracy']:.1%} "
            f"({latest.get('eval_correct', '?')}/{latest['eval_total']}); "
            f"last {len(tail)} shared checkpoints mean {mean(run[i]['eval_accuracy'] for i in tail):.1%}; "
            f"best shared checkpoint {run[best]['eval_accuracy']:.1%} at {best}."
        )
    for label, indices in (("Latest", [shared[-1]]), (f"Last {len(tail)} mean", tail), ("All shared mean", shared)):
        delta = 100 * mean(runs[1][i]["eval_accuracy"] - runs[0][i]["eval_accuracy"] for i in indices)
        lines.append(f"{label} accuracy difference (MC − default): {delta:+.2f} percentage points.")
    lines.extend(["", "Latest shared checkpoint metrics (default / MC):"])
    for key, title, _ in METRICS[2:]:
        values = [metric_value(run[shared[-1]], key) for run in runs]
        lines.append(f"  {title}: {values[0]:.5g} / {values[1]:.5g}")
    lines.extend(["", "Interpretation limits:", *[f"- {note}" for note in notes],
                  "- Aggregate logs do not verify identical eval task IDs or runtime/config/code versions.",
                  "- Repeated checkpoints reuse eval tasks; checkpoint means are descriptive, not independent trials.",
                  "- These two runs alone do not establish a statistically reliable method advantage.",
                  "- Token efficiency measures rollout token usage, not training compute or wall-clock cost.",
                  "- CSV differences use native units (accuracy/rate fractions, not percentage points)."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/train"))
    parser.add_argument("--default-run", default="qwen-bcplus-train", help="Run name or absolute run directory.")
    parser.add_argument("--mc-run", default="qwen-bcplus-compact-value-mc", help="Run name or absolute run directory.")
    parser.add_argument("--output-dir", type=Path, help="Default: <artifacts-dir>/compact_value_mc_comparison")
    parser.add_argument("--last-n", type=int, default=5, help="Trailing shared checkpoints to average (default: 5).")
    args = parser.parse_args()
    if args.last_n < 1:
        parser.error("--last-n must be positive")
    try:
        paths = [args.artifacts_dir / name / "eval_metrics.jsonl" for name in (args.default_run, args.mc_run)]
        runs = [load_metrics(path) for path in paths]
        shared = sorted(set(runs[0]) & set(runs[1]))
        if not shared:
            raise ValueError("The runs have no shared checkpoint iterations")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    notes = comparison_notes(runs, shared)
    for note in notes:
        print(f"Note: {note}", file=sys.stderr)
    output = args.output_dir or args.artifacts_dir / "compact_value_mc_comparison"
    output.mkdir(parents=True, exist_ok=True)
    render_plots(runs, shared, output)
    write_comparison(runs, shared, output)
    summary = summary_text(runs, shared, args.last_n, notes)
    summary += "\nInputs:\n" + "\n".join(str(path.resolve()) for path in paths) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Plots, matched metrics CSV, and summary written to {output}")


if __name__ == "__main__":
    main()
