from __future__ import annotations

"""Compare eval metrics between the compaction-training run and the
no-compaction-loss ablation run.

Both runs share the same 12k context threshold; the only difference is
train_compaction_tokens (default.yaml trains the compaction/summary tokens,
no_compaction_loss.yaml keeps compaction in the rollout but masks its tokens
from the loss). Overlaying their eval curves isolates the behavioral effect of
training compaction. Renders one SVG per selected metric.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Colorblind-safe categorical pair (validated with the dataviz skill's
# validate_palette.js against the white chart surface). Blue = trains
# compaction, violet = no compaction loss; assigned by role, never by rank.
TRAIN_COLOR = "#2a78d6"
ABLATION_COLOR = "#4a3aa7"

TRAIN_RUN = "qwen-bcplus-train"
ABLATION_RUN = "qwen-bcplus-no-compaction-loss"


@dataclasses.dataclass(frozen=True)
class PanelSpec:
    title: str
    ylabel: str
    kind: str  # "percent" | "tokens" | "count" | "ratio"
    source_key: str | None = None  # None → computed malformed rate


METRIC_REGISTRY: dict[str, PanelSpec] = {
    "eval_accuracy": PanelSpec("Eval accuracy", "Accuracy", "percent", "eval_accuracy"),
    "eval_malformed": PanelSpec("Eval malformed rate", "Malformed rate", "percent"),
    "eval_avg_summary_tokens": PanelSpec(
        "Avg summary tokens", "Tokens / episode", "tokens", "eval_avg_summary_tokens"
    ),
    "eval_avg_summary_generated_tokens": PanelSpec(
        "Avg summary generated tokens", "Tokens / episode", "tokens", "eval_avg_summary_generated_tokens"
    ),
    "eval_avg_summary_count": PanelSpec(
        "Avg summary count", "Summaries / episode", "count", "eval_avg_summary_count"
    ),
    "eval_avg_reasoning_generated_tokens": PanelSpec(
        "Avg reasoning generated tokens", "Tokens / episode", "tokens", "eval_avg_reasoning_generated_tokens"
    ),
    "eval_avg_budget_consumed_tokens": PanelSpec(
        "Avg budget consumed tokens", "Tokens / episode", "tokens", "eval_avg_budget_consumed_tokens"
    ),
    "eval_avg_max_prompt_tokens_seen": PanelSpec(
        "Avg max prompt tokens seen", "Tokens", "tokens", "eval_avg_max_prompt_tokens_seen"
    ),
    "eval_avg_search_calls": PanelSpec(
        "Avg search calls", "Calls / episode", "count", "eval_avg_search_calls"
    ),
    "eval_correct_per_1k_budget_consumed_tokens": PanelSpec(
        "Correct per 1k budget tokens", "Correct / 1k tokens", "ratio", "eval_correct_per_1k_budget_consumed_tokens"
    ),
}

DEFAULT_METRICS = [
    "eval_accuracy",
    "eval_malformed",
    "eval_avg_summary_tokens",
    "eval_avg_summary_generated_tokens",
    "eval_avg_summary_count",
    "eval_avg_reasoning_generated_tokens",
    "eval_avg_budget_consumed_tokens",
]


@dataclasses.dataclass(frozen=True)
class SeriesSpec:
    name: str
    short_label: str
    full_label: str
    color: str
    metrics_path: Path


@dataclasses.dataclass
class SeriesData:
    spec: SeriesSpec
    rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eval metrics between the compaction-training run and the no-compaction-loss ablation."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/train",
        help="Directory of training run dirs (default: artifacts/train).",
    )
    parser.add_argument(
        "--train-run",
        default=TRAIN_RUN,
        help=f"Run that trains compaction tokens (default: {TRAIN_RUN}).",
    )
    parser.add_argument(
        "--ablation-run",
        default=ABLATION_RUN,
        help=f"Run that does not train compaction tokens (default: {ABLATION_RUN}).",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=list(METRIC_REGISTRY),
        default=DEFAULT_METRICS,
        help="Eval metrics to plot as panels (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. With multiple metrics, used as a stem template '<stem>_<metric>.<ext>'. Defaults to <artifacts-dir>/compaction_loss_<metric>.svg.",
    )
    return parser.parse_args()


def build_series(args: argparse.Namespace) -> list[SeriesSpec]:
    artifacts_dir = Path(args.artifacts_dir)
    train_path = artifacts_dir / args.train_run / "eval_metrics.jsonl"
    ablation_path = artifacts_dir / args.ablation_run / "eval_metrics.jsonl"
    missing = [str(path) for path in (train_path, ablation_path) if not path.is_file()]
    if missing:
        print(f"Missing eval metrics: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    return [
        SeriesSpec(
            name=args.train_run,
            short_label="compaction trained",
            full_label=f"Compaction trained ({args.train_run})",
            color=TRAIN_COLOR,
            metrics_path=train_path,
        ),
        SeriesSpec(
            name=args.ablation_run,
            short_label="compaction not trained",
            full_label=f"Compaction not trained ({args.ablation_run})",
            color=ABLATION_COLOR,
            metrics_path=ablation_path,
        ),
    ]


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metric records found in {path}")
    rows.sort(key=lambda record: int(record.get("iteration", 0)))
    return rows


def extract_value(record: dict[str, Any], spec: PanelSpec) -> float:
    if spec.source_key is not None:
        return float(record.get(spec.source_key, 0.0))
    malformed = float(record.get("eval_malformed", 0.0))
    total = float(record.get("eval_total", 0.0))
    return malformed / total if total > 0 else 0.0


def _add_end_labels(ax: Any, series: list[tuple[Any, float, float, str]]) -> None:
    """Direct-label each series at its last point, staggering vertically when finals cluster."""
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    offset_pts = 0.0
    prev_y: float | None = None
    for _line, x, y, label in sorted(series, key=lambda item: item[2], reverse=True):
        if prev_y is not None and abs(y - prev_y) < 0.04 * y_span:
            offset_pts += 6.0
        else:
            offset_pts = 0.0
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(4, offset_pts),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="medium",
            color="#374151",
        )
        prev_y = y


def render_metric_figure(series: list[SeriesData], metric_key: str, output_path: Path) -> None:
    """Render one metric as a standalone SVG with both runs' curves overlaid."""
    spec = METRIC_REGISTRY[metric_key]

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#f9fafb")
    fig.suptitle(
        f"Compaction loss training comparison — {spec.title}",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.set_facecolor("#ffffff")

    x_min = min(int(record.get("iteration", 0)) for data in series for record in data.rows)
    x_max = max(int(record.get("iteration", 0)) for data in series for record in data.rows)
    span = x_max - x_min + 1

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    all_values: list[float] = []
    panel_series: list[tuple[Any, float, float, str]] = []

    for data in series:
        if spec.source_key is not None and any(
            spec.source_key not in record for record in data.rows
        ):
            print(
                f"Warning: {data.spec.name}: '{spec.source_key}' missing from some eval records; treating as 0.0",
                file=sys.stderr,
            )
        xs = [int(record.get("iteration", 0)) for record in data.rows]
        ys = [extract_value(record, spec) for record in data.rows]
        (line,) = ax.plot(
            xs,
            ys,
            color=data.spec.color,
            linewidth=2.5,
            marker="o",
            markerfacecolor="#ffffff",
            markeredgecolor=data.spec.color,
            markeredgewidth=1.5,
            markersize=5,
            zorder=5,
        )
        all_values.extend(ys)
        if len(ys) >= 2:
            panel_series.append((line, xs[-1], ys[-1], data.spec.short_label))
        legend_handles.append(line)
        legend_labels.append(data.spec.full_label)

    # Y axis: limits and formatter per metric kind
    if spec.kind == "percent":
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    else:
        y_max = max(all_values) * 1.08
        if y_max <= 0:
            y_max = 1.0
        ax.set_ylim(0.0, y_max)
        if spec.kind == "tokens":
            if y_max >= 10_000:
                formatter = lambda v, _, y_max=y_max: f"{v / 1000:g}k"  # noqa: E731
            else:
                formatter = lambda v, _, y_max=y_max: f"{v:.0f}"  # noqa: E731
        elif spec.kind == "count":
            formatter = lambda v, _, y_max=y_max: f"{v:.1f}" if y_max < 10 else f"{v:.0f}"  # noqa: E731
        else:  # ratio
            formatter = lambda v, _, y_max=y_max: f"{v:.3f}"  # noqa: E731
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))

    ax.set_xlim(x_min - 0.5, x_max + 1.8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1 if span <= 20 else 5))
    ax.tick_params(axis="x", labelsize=10, colors="#374151")
    ax.tick_params(axis="y", labelsize=11, colors="#6b7280")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="#e5e7eb", alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(1.5)
    ax.set_xlabel("Iteration", fontsize=14, fontweight="medium", color="#4b5563", labelpad=8)
    ax.set_ylabel(spec.ylabel, fontsize=14, fontweight="medium", color="#4b5563", labelpad=8)

    _add_end_labels(ax, panel_series)

    # Figure-level legend below the axes (full labels); short direct end-labels
    # carry identity inside the plot, so color is never the only encoding.
    legend = fig.legend(
        legend_handles,
        legend_labels,
        ncol=len(legend_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout(pad=2)
    fig.savefig(
        str(output_path),
        dpi=100,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)


def metric_output_path(args: argparse.Namespace, metric: str, metric_count: int) -> Path:
    if args.output:
        base = Path(args.output)
        if metric_count == 1:
            return base
        return base.with_name(f"{base.stem}_{metric}{base.suffix}")
    return Path(args.artifacts_dir) / f"compaction_loss_{metric}.svg"


def main() -> None:
    args = parse_args()
    specs = build_series(args)
    metrics = list(dict.fromkeys(args.metric))  # dedupe, preserve order
    series = [SeriesData(spec=spec, rows=load_metrics(spec.metrics_path)) for spec in specs]
    for spec in specs:
        print(f"{spec.name}: {spec.full_label}")
    for metric in metrics:
        output_path = metric_output_path(args, metric, len(metrics))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_metric_figure(series, metric, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
