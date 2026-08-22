from __future__ import annotations

"""Compare eval metrics across training runs with different compaction thresholds.

Discovers runs under artifacts/train (one eval_metrics.jsonl per run), joins each
run to its config under configs/train via experiment.name to recover the runtime
compaction threshold, and renders one SVG per selected metric with all runs'
curves overlaid.
"""

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yaml

# Colorblind-safe categorical palette (validated with the dataviz skill's
# validate_palette.js against the white chart surface). Assigned by compaction
# threshold ascending: 6k=blue, 12k=aqua, 24k=magenta, none=violet.
PALETTE = ["#2a78d6", "#1baf7a", "#e87ba4", "#4a3aa7"]

# Sentinel used by no_compact_32k.yaml to disable runtime compaction.
DISABLED_THRESHOLD = 10**9


@dataclasses.dataclass(frozen=True)
class PanelSpec:
    title: str
    ylabel: str
    kind: str  # "percent" | "tokens" | "count" | "ratio"
    source_key: str | None = None  # None → computed malformed rate


METRIC_REGISTRY: dict[str, PanelSpec] = {
    "eval_accuracy": PanelSpec("Eval accuracy", "Accuracy", "percent", "eval_accuracy"),
    "eval_malformed": PanelSpec("Eval malformed rate", "Malformed rate", "percent"),
    "eval_avg_budget_consumed_tokens": PanelSpec(
        "Avg budget consumed tokens", "Tokens", "tokens", "eval_avg_budget_consumed_tokens"
    ),
    "eval_avg_reasoning_generated_tokens": PanelSpec(
        "Avg reasoning generated tokens", "Tokens", "tokens", "eval_avg_reasoning_generated_tokens"
    ),
    "eval_avg_summary_tokens": PanelSpec(
        "Avg summary tokens", "Tokens", "tokens", "eval_avg_summary_tokens"
    ),
    "eval_avg_summary_count": PanelSpec(
        "Avg summary count", "Summaries / episode", "count", "eval_avg_summary_count"
    ),
    "eval_avg_max_prompt_tokens_seen": PanelSpec(
        "Avg max prompt tokens seen", "Tokens", "tokens", "eval_avg_max_prompt_tokens_seen"
    ),
    "eval_correct_per_1k_budget_consumed_tokens": PanelSpec(
        "Correct per 1k budget tokens", "Correct / 1k tokens", "ratio", "eval_correct_per_1k_budget_consumed_tokens"
    ),
    "eval_avg_search_calls": PanelSpec(
        "Avg search calls", "Calls / episode", "count", "eval_avg_search_calls"
    ),
}

DEFAULT_METRICS = [
    "eval_accuracy",
    "eval_malformed",
    "eval_avg_budget_consumed_tokens",
    "eval_avg_reasoning_generated_tokens",
    "eval_avg_summary_tokens",
    "eval_avg_summary_count",
]


@dataclasses.dataclass
class ConfigInfo:
    threshold: int | None
    max_tokens: int | None


@dataclasses.dataclass
class RunInfo:
    name: str
    threshold: int | None
    max_tokens: int | None
    short_label: str
    full_label: str
    metrics_path: Path
    color: str = ""


@dataclasses.dataclass
class RunData:
    run: RunInfo
    rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eval metrics across training runs with different compaction thresholds."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/train",
        help="Directory of training run dirs (default: artifacts/train).",
    )
    parser.add_argument(
        "--configs-dir",
        default="configs/train",
        help="Directory of training configs; join key is experiment.name (default: configs/train).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. With multiple metrics, used as a stem template '<stem>_<metric>.<ext>'. Defaults to <artifacts-dir>/comparison_<metric>.svg.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=list(METRIC_REGISTRY),
        default=DEFAULT_METRICS,
        help="Eval metrics to plot as panels (default: %(default)s).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run directory names to include (default: all discovered).",
    )
    return parser.parse_args()


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _fmt_k(tokens: int) -> str:
    return f"{int(tokens / 1000)}k"


def load_configs(configs_dir: Path) -> dict[str, ConfigInfo]:
    configs: dict[str, ConfigInfo] = {}
    if not configs_dir.is_dir():
        return configs
    for path in sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — tolerate a bad config, keep scanning
            print(f"Warning: skipping unparseable config {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(doc, dict):
            continue
        experiment = doc.get("experiment") or {}
        runtime = doc.get("runtime") or {}
        name = experiment.get("name")
        if not name:
            print(f"Warning: config {path} has no experiment.name; skipping", file=sys.stderr)
            continue
        configs[name] = ConfigInfo(
            threshold=_as_int(runtime.get("context_threshold_tokens")),
            max_tokens=_as_int(runtime.get("max_context_tokens")),
        )
    return configs


def describe_run(run_dir: Path, configs: dict[str, ConfigInfo]) -> RunInfo:
    name = run_dir.name
    metrics_path = run_dir / "eval_metrics.jsonl"
    config = configs.get(name)
    if config is None or config.threshold is None:
        print(
            f"Warning: no config found with experiment.name '{name}'; using directory name as label",
            file=sys.stderr,
        )
        return RunInfo(
            name=name,
            threshold=None,
            max_tokens=None,
            short_label=name,
            full_label=name,
            metrics_path=metrics_path,
        )
    threshold = config.threshold
    if threshold >= DISABLED_THRESHOLD:
        short = "none"
        full = (
            f"No compaction (max {_fmt_k(config.max_tokens)})"
            if config.max_tokens is not None
            else "No compaction"
        )
    else:
        short = f"{threshold / 1000:g}k"
        full = f"{short} (max {_fmt_k(config.max_tokens)})" if config.max_tokens is not None else short
    return RunInfo(
        name=name,
        threshold=threshold,
        max_tokens=config.max_tokens,
        short_label=short,
        full_label=full,
        metrics_path=metrics_path,
    )


def discover_runs(
    artifacts_dir: Path, configs_dir: Path, run_filter: list[str] | None = None
) -> list[RunInfo]:
    configs = load_configs(configs_dir)
    run_dirs = sorted(path.parent for path in sorted(artifacts_dir.glob("*/eval_metrics.jsonl")))
    runs = [describe_run(run_dir, configs) for run_dir in run_dirs]
    # Threshold ascending, disabled last, then by name. Colors are assigned from
    # this full ordering, so a run keeps its color even when a filter drops its
    # neighbors — color follows the threshold, never the selection rank.
    runs.sort(key=lambda run: (run.threshold if run.threshold is not None else math.inf, run.name))
    if len(runs) > len(PALETTE):
        print(
            f"Warning: {len(runs)} runs but palette defines {len(PALETTE)} colors; colors will repeat",
            file=sys.stderr,
        )
    for idx, run in enumerate(runs):
        run.color = PALETTE[idx % len(PALETTE)]
    if run_filter:
        wanted = set(run_filter)
        for name in wanted - {run.name for run in runs}:
            print(f"Warning: run '{name}' not found under {artifacts_dir}", file=sys.stderr)
        runs = [run for run in runs if run.name in wanted]
    return runs


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


def render_metric_figure(runs: list[RunData], metric_key: str, output_path: Path) -> None:
    """Render one metric as a standalone SVG with all runs' curves overlaid."""
    spec = METRIC_REGISTRY[metric_key]

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("#f9fafb")
    fig.suptitle(
        f"Compaction threshold comparison — {spec.title}",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.set_facecolor("#ffffff")

    x_min = min(int(record.get("iteration", 0)) for data in runs for record in data.rows)
    x_max = max(int(record.get("iteration", 0)) for data in runs for record in data.rows)
    span = x_max - x_min + 1

    warned: set[tuple[str, str]] = set()
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    all_values: list[float] = []
    panel_series: list[tuple[Any, float, float, str]] = []

    for data in runs:
        run = data.run
        if spec.source_key is not None and any(
            spec.source_key not in record for record in data.rows
        ):
            warn_key = (run.name, spec.source_key)
            if warn_key not in warned:
                warned.add(warn_key)
                print(
                    f"Warning: {run.name}: '{spec.source_key}' missing from some eval records; treating as 0.0",
                    file=sys.stderr,
                )
        xs = [int(record.get("iteration", 0)) for record in data.rows]
        ys = [extract_value(record, spec) for record in data.rows]
        (line,) = ax.plot(
            xs,
            ys,
            color=run.color,
            linewidth=2.5,
            marker="o",
            markerfacecolor="#ffffff",
            markeredgecolor=run.color,
            markeredgewidth=1.5,
            markersize=5,
            zorder=5,
        )
        all_values.extend(ys)
        if len(ys) >= 2:
            panel_series.append((line, xs[-1], ys[-1], run.short_label))
        legend_handles.append(line)
        legend_labels.append(run.full_label)

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
    return Path(args.artifacts_dir) / f"comparison_{metric}.svg"


def main() -> None:
    args = parse_args()
    runs = discover_runs(Path(args.artifacts_dir), Path(args.configs_dir), args.runs)
    if not runs:
        print(f"No training runs with eval_metrics.jsonl found under {args.artifacts_dir}", file=sys.stderr)
        sys.exit(1)
    metrics = list(dict.fromkeys(args.metric))  # dedupe, preserve order
    run_data = [RunData(run=run, rows=load_metrics(run.metrics_path)) for run in runs]
    for run in runs:
        print(f"{run.name}: {run.full_label}")
    for metric in metrics:
        output_path = metric_output_path(args, metric, len(metrics))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_metric_figure(run_data, metric, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
