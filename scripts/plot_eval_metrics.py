from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval accuracy curve from eval_metrics.jsonl.")
    parser.add_argument(
        "eval_metrics",
        help="Path to artifacts/train/<experiment>/eval_metrics.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to eval_metrics.svg next to the input.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "iteration": int(record["iteration"]),
                    "accuracy": float(record["eval_accuracy"]),
                    "correct": int(record.get("eval_correct", 0)),
                    "total": int(record.get("eval_total", 0)),
                    "malformed": int(record.get("eval_malformed", 0)),
                    "parse_errors": int(record.get("eval_parse_errors", 0)),
                }
            )
    if not rows:
        raise ValueError(f"No metric records found in {path}")
    rows.sort(key=lambda r: r["iteration"])
    return rows


def render_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    iterations = [r["iteration"] for r in rows]
    accuracies = [r["accuracy"] for r in rows]

    y_max = min(max(accuracies) * 1.12, 1.0)
    y_min = 0.0

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#f9fafb")
    ax.set_facecolor("#ffffff")

    # Area under curve
    ax.fill_between(iterations, accuracies, alpha=0.18, color="#14b8a6", linewidth=0)

    # Main curve line with markers
    (line,) = ax.plot(
        iterations,
        accuracies,
        color="#0f766e",
        linewidth=2.5,
        marker="o",
        markerfacecolor="#ffffff",
        markeredgecolor="#0f766e",
        markeredgewidth=2.5,
        markersize=10,
        zorder=5,
        label="Eval accuracy",
    )

    # Percentage labels above each data point
    for x, y in zip(iterations, accuracies):
        ax.annotate(
            f"{y:.0%}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#0f766e",
            fontfamily="sans-serif",
        )

    # Axes limits
    ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    ax.set_ylim(y_min, y_max)

    # X-axis: show every iteration
    ax.set_xticks(iterations)
    ax.tick_params(axis="x", labelsize=10, colors="#374151")

    # Y-axis: grid and formatting
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(axis="y", labelsize=11, colors="#6b7280")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="#e5e7eb", alpha=0.8)
    ax.set_axisbelow(True)

    # Axis labels
    ax.set_xlabel("Iteration", fontsize=14, fontweight="medium", color="#4b5563", labelpad=8)
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="medium", color="#4b5563", labelpad=8)

    # Title
    ax.set_title(
        "Evaluation Accuracy Curve",
        fontsize=18,
        fontweight="bold",
        color="#111827",
        pad=18,
    )

    # Legend — placed in lower right, inside the axes but below the curve data
    ax.legend(
        loc="lower right",
        framealpha=0.95,
        edgecolor="#d1d5db",
        facecolor="#ffffff",
        fontsize=11,
    )

    # Border styling
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(1.5)

    fig.tight_layout(pad=2)
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.eval_metrics)
    output_path = Path(args.output) if args.output else metrics_path.with_suffix(".svg")
    rows = load_metrics(metrics_path)
    render_plot(rows, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
