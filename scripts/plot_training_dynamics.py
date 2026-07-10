from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PLOT_CHOICES = ["all", "tokens", "efficiency", "behavior", "errors", "context", "reward"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training-dynamics charts from eval_metrics.jsonl and step_metrics.jsonl as SVGs."
    )
    parser.add_argument(
        "eval_metrics",
        help="Path to artifacts/train/<experiment>/eval_metrics.jsonl",
    )
    parser.add_argument(
        "step_metrics",
        help="Path to artifacts/train/<experiment>/step_metrics.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output SVGs. Defaults next to eval_metrics.",
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        choices=PLOT_CHOICES,
        default=["all"],
        help="Which plots to generate (default: all).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "iteration": int(rec["iteration"]),
                    "accuracy": float(rec["eval_accuracy"]),
                    "correct": int(rec.get("eval_correct", 0)),
                    "total": int(rec.get("eval_total", 0)),
                    "malformed": int(rec.get("eval_malformed", 0)),
                    "parse_errors": int(rec.get("eval_parse_errors", 0)),
                    "avg_reasoning_tokens": float(rec.get("eval_avg_reasoning_generated_tokens", 0)),
                    "avg_summary_tokens": float(rec.get("eval_avg_summary_generated_tokens", 0)),
                    "avg_forced_answer_tokens": float(rec.get("eval_avg_forced_answer_generated_tokens", 0)),
                    "avg_tool_result_tokens": float(rec.get("eval_avg_tool_result_tokens", 0)),
                    "avg_total_tokens": float(rec.get("eval_avg_total_generated_tokens", 0)),
                    "avg_max_prompt_tokens": float(rec.get("eval_avg_max_prompt_tokens_seen", 0)),
                    "max_prompt_tokens": float(rec.get("eval_max_prompt_tokens_seen", 0)),
                    "avg_search_calls": float(rec.get("eval_avg_search_calls", 0)),
                    "avg_document_calls": float(rec.get("eval_avg_document_calls", 0)),
                    "avg_summary_count": float(rec.get("eval_avg_summary_count", 0)),
                    "correct_per_1k_reasoning": float(rec.get("eval_correct_per_1k_reasoning_tokens", 0)),
                    "correct_per_1k_total": float(rec.get("eval_correct_per_1k_total_generated_tokens", 0)),
                }
            )
    if not rows:
        raise ValueError(f"No records in {path}")
    rows.sort(key=lambda r: r["iteration"])
    return rows


def load_step_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "step": idx,
                    "mean_reward": float(rec.get("mean_reward", 0)),
                    "tokens_per_sec": float(rec.get("verl_fsdp/effective_train_tokens_per_second", 0)),
                    "train_tokens": int(rec.get("verl_fsdp/train_tokens", 0)),
                    "sample_count": int(rec.get("sample_count", 0)),
                    "update_seconds": float(rec.get("verl_fsdp/update_seconds", 0)),
                }
            )
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

WIDTH = 960
HEIGHT = 600
PAD = 70


def scale_x(value: float, *, min_x: float, max_x: float) -> float:
    span = max(max_x - min_x, 1)
    return PAD + ((value - min_x) / span) * (WIDTH - 2 * PAD)


def scale_y(value: float, *, min_y: float, max_y: float) -> float:
    span = max(max_y - min_y, 0.01)
    return HEIGHT - PAD - ((value - min_y) / span) * (HEIGHT - 2 * PAD)


def polyline_points(
    xs: list[float], ys: list[float], *, x_min: float, x_max: float, y_min: float, y_max: float
) -> str:
    pts = []
    for x, y in zip(xs, ys):
        sx = scale_x(x, min_x=x_min, max_x=x_max)
        sy = scale_y(y, min_y=y_min, max_y=y_max)
        pts.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(pts)


def area_points(
    xs: list[float], ys: list[float], *, x_min: float, x_max: float, y_min: float, y_max: float
) -> str:
    pts = list(polyline_points(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max).split())
    baseline_y = scale_y(y_min, min_y=y_min, max_y=y_max)
    pts.append(f"{scale_x(x_max, min_x=x_min, max_x=x_max):.1f},{baseline_y:.1f}")
    pts.append(f"{scale_x(x_min, min_x=x_min, max_x=x_max):.1f},{baseline_y:.1f}")
    return " ".join(pts)


def grid_lines(*, y_min: float, y_max: float, x_min: float | None = None, x_max: float | None = None) -> str:
    """Horizontal grid lines at 0, 0.25, 0.5, 0.75, 1.0 of y-range."""
    lines: list[str] = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = y_min + frac * (y_max - y_min)
        y = scale_y(val, min_y=y_min, max_y=y_max)
        lines.append(
            f'<line x1="{PAD}" y1="{y:.1f}" x2="{WIDTH - PAD}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        lines.append(
            f'<text x="{PAD - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#6b7280">'
            f"{_fmt_y(val)}</text>"
        )
    return "\n  ".join(lines)


def _fmt_y(val: float) -> str:
    if val >= 1000:
        return f"{val/1000:.0f}k"
    if val >= 1:
        return f"{val:.1f}"
    return f"{val:.3f}"


def x_axis_labels(xs: list[float], *, x_min: float, x_max: float, label_fn=None) -> str:
    if label_fn is None:
        label_fn = lambda v: str(int(v))
    labels: list[str] = []
    for x in xs:
        sx = scale_x(x, min_x=x_min, max_x=x_max)
        labels.append(
            f'<text x="{sx:.1f}" y="{HEIGHT - PAD + 22}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#374151">'
            f'{label_fn(x)}</text>'
        )
    return "\n  ".join(labels)


def markers(
    xs: list[float], ys: list[float], *, x_min: float, x_max: float, y_min: float, y_max: float,
    color: str, label_fn=None
) -> tuple[str, str]:
    m_circles: list[str] = []
    m_labels: list[str] = []
    for x, y in zip(xs, ys):
        sx = scale_x(x, min_x=x_min, max_x=x_max)
        sy = scale_y(y, min_y=y_min, max_y=y_max)
        m_circles.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="#ffffff" stroke="{color}" stroke-width="2"/>'
        )
        if label_fn:
            lbl = label_fn(y)
            m_labels.append(
                f'<text x="{sx:.1f}" y="{sy - 12:.1f}" text-anchor="middle" '
                f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="10" fill="{color}" font-weight="600">'
                f'{lbl}</text>'
            )
    return "\n  ".join(m_circles), "\n  ".join(m_labels)


def svg_header(title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <defs>
    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.20"/>
      <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.0"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="#f9fafb" rx="8" ry="8"/>
  <rect x="10" y="10" width="{WIDTH - 20}" height="{HEIGHT - 20}" fill="#ffffff" rx="6" ry="6" stroke="#e5e7eb" stroke-width="1"/>
  <text x="{WIDTH / 2:.1f}" y="42" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="20" fill="#111827" font-weight="600">
    {title}
  </text>"""


def svg_footer(x_label: str, y_label: str) -> str:
    return f"""  <line x1="{PAD}" y1="{HEIGHT - PAD}" x2="{WIDTH - PAD}" y2="{HEIGHT - PAD}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>
  <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{HEIGHT - PAD}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>
  <text x="{WIDTH / 2:.1f}" y="{HEIGHT - 6}" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#4b5563" font-weight="500">
    {x_label}
  </text>
  <text x="18" y="{HEIGHT / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {HEIGHT / 2:.1f})" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#4b5563" font-weight="500">
    {y_label}
  </text>
</svg>"""


def legend_horizontal(items: list[tuple[str, str]]) -> str:
    """Horizontal legend centered below the x-axis, outside the plot area."""
    y = HEIGHT - PAD + 38
    swatch_w = 16.0
    text_pad = 6.0
    item_gap = 28.0
    char_w = 6.8

    total_w = sum(swatch_w + text_pad + len(label) * char_w for _, label in items)
    total_w += item_gap * (len(items) - 1)

    x = (WIDTH - total_w) / 2.0
    lines: list[str] = []
    for color, label in items:
        label_w = len(label) * char_w
        lines.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + swatch_w:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        lines.append(
            f'<circle cx="{x + swatch_w / 2:.1f}" cy="{y:.1f}" r="3" fill="#ffffff" stroke="{color}" stroke-width="1.5"/>'
        )
        x += swatch_w + text_pad
        lines.append(
            f'<text x="{x:.1f}" y="{y + 5:.1f}" font-family="ui-sans-serif, system-ui, Arial, sans-serif" '
            f'font-size="12" fill="#374151">{label}</text>'
        )
        x += label_w + item_gap
    return "\n  ".join(lines)


# ---------------------------------------------------------------------------
# Plot renderers
# ---------------------------------------------------------------------------


def plot_tokens(rows: list[dict[str, Any]]) -> str:
    title = "Token Usage per Episode"
    xs = [r["iteration"] for r in rows]
    x_min, x_max = min(xs), max(xs)

    series = [
        ("reasoning", "#2563eb", [r["avg_reasoning_tokens"] for r in rows]),
        ("summary", "#dc2626", [r["avg_summary_tokens"] for r in rows]),
        ("tool results", "#16a34a", [r["avg_tool_result_tokens"] for r in rows]),
        ("forced answer", "#9333ea", [r["avg_forced_answer_tokens"] for r in rows]),
    ]

    all_y = [v for _, _, ys in series for v in ys]
    y_max = max(all_y) * 1.10
    y_min = 0.0

    svg = [svg_header(title)]
    svg.append(f"  <!-- Grid + Y labels -->\n  {grid_lines(y_min=y_min, y_max=y_max)}")

    for label, color, ys in series:
        pts = polyline_points(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        svg.append(
            f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    svg.append(f"  <!-- X-axis labels -->\n  {x_axis_labels(xs, x_min=x_min, x_max=x_max)}")
    svg.append(
        f"  {legend_horizontal([(c, l) for l, c, _ in series])}"
    )
    svg.append(svg_footer("Iteration", "Avg Tokens / Episode"))
    return "\n".join(svg)


def plot_efficiency(rows: list[dict[str, Any]]) -> str:
    title = "Token Efficiency (Correct Answers per 1k Tokens)"
    xs = [r["iteration"] for r in rows]
    x_min, x_max = min(xs), max(xs)

    series = [
        ("per 1k reasoning", "#2563eb", [r["correct_per_1k_reasoning"] for r in rows]),
        ("per 1k total", "#dc2626", [r["correct_per_1k_total"] for r in rows]),
    ]

    all_y = [v for _, _, ys in series for v in ys]
    y_max = max(all_y) * 1.12
    y_min = max(min(all_y) * 0.85, 0.0)

    svg = [svg_header(title)]
    svg.append(f"  <!-- Grid + Y labels -->\n  {grid_lines(y_min=y_min, y_max=y_max)}")

    for label, color, ys in series:
        pts = polyline_points(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        svg.append(
            f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        # markers on every point
        m_circ, _ = markers(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, color=color)
        svg.append(f"  {m_circ}")

    svg.append(f"  <!-- X-axis labels -->\n  {x_axis_labels(xs, x_min=x_min, x_max=x_max)}")
    svg.append(
        f"  {legend_horizontal([(c, l) for l, c, _ in series])}"
    )
    svg.append(svg_footer("Iteration", "Correct / 1k Tokens"))
    return "\n".join(svg)


def plot_behavior(rows: list[dict[str, Any]]) -> str:
    title = "Agent Behavior"
    xs = [r["iteration"] for r in rows]
    x_min, x_max = min(xs), max(xs)

    series = [
        ("search calls", "#2563eb", [r["avg_search_calls"] for r in rows]),
        ("summary count", "#dc2626", [r["avg_summary_count"] for r in rows]),
        ("document calls", "#16a34a", [r["avg_document_calls"] for r in rows]),
    ]

    all_y = [v for _, _, ys in series for v in ys]
    y_max = max(all_y) * 1.15
    y_min = 0.0

    svg = [svg_header(title)]
    svg.append(f"  <!-- Grid + Y labels -->\n  {grid_lines(y_min=y_min, y_max=y_max)}")

    for label, color, ys in series:
        pts = polyline_points(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        svg.append(
            f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    svg.append(f"  <!-- X-axis labels -->\n  {x_axis_labels(xs, x_min=x_min, x_max=x_max)}")
    svg.append(
        f"  {legend_horizontal([(c, l) for l, c, _ in series])}"
    )
    svg.append(svg_footer("Iteration", "Avg Count / Episode"))
    return "\n".join(svg)


def plot_errors(rows: list[dict[str, Any]]) -> str:
    title = "Error Rates"
    xs = [r["iteration"] for r in rows]
    x_min, x_max = min(xs), max(xs)

    malformed_rates = [r["malformed"] / r["total"] for r in rows]
    parse_rates = [r["parse_errors"] / r["total"] for r in rows]

    series = [
        ("malformed rate", "#dc2626", malformed_rates),
        ("parse error rate", "#f59e0b", parse_rates),
    ]

    all_y = [v for _, _, ys in series for v in ys]
    y_max = max(all_y) * 1.20
    y_min = 0.0

    svg = [svg_header(title)]
    svg.append(f"  <!-- Grid + Y labels -->\n  {grid_lines(y_min=y_min, y_max=y_max)}")

    for label, color, ys in series:
        pts = polyline_points(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        svg.append(
            f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    svg.append(f"  <!-- X-axis labels -->\n  {x_axis_labels(xs, x_min=x_min, x_max=x_max)}")
    svg.append(
        f"  {legend_horizontal([(c, l) for l, c, _ in series])}"
    )
    svg.append(svg_footer("Iteration", "Rate"))
    return "\n".join(svg)


def plot_context(rows: list[dict[str, Any]]) -> str:
    title = "Context Window Utilization"
    xs = [r["iteration"] for r in rows]
    x_min, x_max = min(xs), max(xs)

    series = [
        ("avg max prompt", "#2563eb", [r["avg_max_prompt_tokens"] for r in rows]),
        ("peak max prompt", "#dc2626", [r["max_prompt_tokens"] for r in rows]),
    ]

    all_y = [v for _, _, ys in series for v in ys]
    y_max = max(all_y) * 1.12
    y_min = 0.0

    svg = [svg_header(title)]
    svg.append(f"  <!-- Grid + Y labels -->\n  {grid_lines(y_min=y_min, y_max=y_max)}")

    for label, color, ys in series:
        pts = polyline_points(xs, ys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        svg.append(
            f'  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    svg.append(f"  <!-- X-axis labels -->\n  {x_axis_labels(xs, x_min=x_min, x_max=x_max)}")
    svg.append(
        f"  {legend_horizontal([(c, l) for l, c, _ in series])}"
    )
    svg.append(svg_footer("Iteration", "Tokens"))
    return "\n".join(svg)


def plot_reward(eval_rows: list[dict[str, Any]], step_rows: list[dict[str, Any]]) -> str:
    """Dual-Y-axis: mean_reward (left) and tokens/sec (right)."""
    title = "Training Reward and Throughput"

    xs = [r["step"] for r in step_rows]
    x_min, x_max = min(xs), max(xs)

    reward_ys = [r["mean_reward"] for r in step_rows]
    tps_ys = [r["tokens_per_sec"] for r in step_rows]

    # Reward axis (left)
    r_min = min(reward_ys) * 0.90
    r_max = max(reward_ys) * 1.10

    # Throughput axis (right)  -- map to the same canvas space
    t_min = min(tps_ys) * 0.90
    t_max = max(tps_ys) * 1.10

    svg = [svg_header(title)]

    # Reward grid (left axis)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = r_min + frac * (r_max - r_min)
        y = scale_y(val, min_y=r_min, max_y=r_max)
        svg.append(
            f'  <line x1="{PAD}" y1="{y:.1f}" x2="{WIDTH - PAD}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        svg.append(
            f'  <text x="{PAD - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#2563eb">'
            f'{val:.2f}</text>'
        )

    # Reward line
    pts = polyline_points(xs, reward_ys, x_min=x_min, x_max=x_max, y_min=r_min, y_max=r_max)
    svg.append(
        f'  <polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    # Throughput line
    pts_t = polyline_points(xs, tps_ys, x_min=x_min, x_max=x_max, y_min=t_min, y_max=t_max)
    svg.append(
        f'  <polyline points="{pts_t}" fill="none" stroke="#16a34a" stroke-width="2.5" '
        f'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="6,3"/>'
    )

    # Right-axis labels for throughput
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = t_min + frac * (t_max - t_min)
        y = scale_y(val, min_y=t_min, max_y=t_max)
        svg.append(
            f'  <text x="{WIDTH - PAD + 10}" y="{y + 4:.1f}" text-anchor="start" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#16a34a">'
            f'{val:.0f}</text>'
        )

    svg.append(f"  <!-- X-axis labels -->\n  {x_axis_labels(xs, x_min=x_min, x_max=x_max)}")
    svg.append(
        f"  {legend_horizontal([('#2563eb', 'mean reward'), ('#16a34a', 'tokens/sec')])}"
    )

    # Footer with dual axis labels
    svg.append(f"""  <line x1="{PAD}" y1="{HEIGHT - PAD}" x2="{WIDTH - PAD}" y2="{HEIGHT - PAD}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>
  <line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{HEIGHT - PAD}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>
  <text x="{WIDTH / 2:.1f}" y="{HEIGHT - 6}" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#4b5563" font-weight="500">
    Training Step
  </text>
  <text x="18" y="{HEIGHT / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {HEIGHT / 2:.1f})" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#2563eb" font-weight="500">
    Mean Reward
  </text>
  <text x="{WIDTH - 6}" y="{HEIGHT / 2:.1f}" text-anchor="middle" transform="rotate(-90 {WIDTH - 6} {HEIGHT / 2:.1f})" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#16a34a" font-weight="500">
    Tokens / sec
  </text>
</svg>""")
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PLOT_REGISTRY = {
    "tokens": ("training_tokens.svg", plot_tokens, True),
    "efficiency": ("training_efficiency.svg", plot_efficiency, True),
    "behavior": ("training_behavior.svg", plot_behavior, True),
    "errors": ("training_errors.svg", plot_errors, True),
    "context": ("training_context.svg", plot_context, True),
    "reward": ("training_reward.svg", plot_reward, False),
}


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_metrics)
    step_path = Path(args.step_metrics)
    output_dir = Path(args.output_dir) if args.output_dir else eval_path.parent

    plots_to_run = set(args.plot)
    if "all" in plots_to_run:
        plots_to_run = set(PLOT_REGISTRY.keys())

    eval_rows = load_eval_metrics(eval_path)
    step_rows = load_step_metrics(step_path)

    for plot_name in sorted(plots_to_run):
        filename, render_fn, needs_eval = PLOT_REGISTRY[plot_name]
        out_path = output_dir / filename
        if needs_eval:
            svg = render_fn(eval_rows)
        else:
            svg = render_fn(eval_rows, step_rows)
        out_path.write_text(svg, encoding="utf-8")
        print(out_path)


if __name__ == "__main__":
    main()
