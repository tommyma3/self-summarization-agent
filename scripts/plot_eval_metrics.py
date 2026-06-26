from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval accuracy curve from eval_metrics.jsonl as an SVG.")
    parser.add_argument(
        "eval_metrics",
        help="Path to artifacts/train/<experiment>/eval_metrics.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path. Defaults to eval_metrics.svg next to the input.",
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


def _scale_x(value: int, *, min_iter: int, max_iter: int, width: int, pad: int) -> float:
    span = max(max_iter - min_iter, 1)
    return pad + ((value - min_iter) / span) * (width - 2 * pad)


def _scale_y(value: float, *, height: int, pad: int, y_min: float, y_max: float) -> float:
    span = max(y_max - y_min, 0.01)
    return height - pad - ((value - y_min) / span) * (height - 2 * pad)


def _line_points(rows: list[dict[str, Any]], *, width: int, height: int, pad: int, y_min: float, y_max: float) -> str:
    min_iter = rows[0]["iteration"]
    max_iter = rows[-1]["iteration"]
    points = []
    for row in rows:
        x = _scale_x(row["iteration"], min_iter=min_iter, max_iter=max_iter, width=width, pad=pad)
        y = _scale_y(row["accuracy"], height=height, pad=pad, y_min=y_min, y_max=y_max)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _area_points(rows: list[dict[str, Any]], *, width: int, height: int, pad: int, y_min: float, y_max: float) -> str:
    min_iter = rows[0]["iteration"]
    max_iter = rows[-1]["iteration"]
    baseline_y = _scale_y(y_min, height=height, pad=pad, y_min=y_min, y_max=y_max)
    points = []
    for row in rows:
        x = _scale_x(row["iteration"], min_iter=min_iter, max_iter=max_iter, width=width, pad=pad)
        y = _scale_y(row["accuracy"], height=height, pad=pad, y_min=y_min, y_max=y_max)
        points.append(f"{x:.1f},{y:.1f}")
    last_x = _scale_x(max_iter, min_iter=min_iter, max_iter=max_iter, width=width, pad=pad)
    first_x = _scale_x(min_iter, min_iter=min_iter, max_iter=max_iter, width=width, pad=pad)
    points.append(f"{last_x:.1f},{baseline_y:.1f}")
    points.append(f"{first_x:.1f},{baseline_y:.1f}")
    return " ".join(points)


def _grid_lines(*, width: int, height: int, pad: int, y_min: float, y_max: float) -> str:
    lines = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = y_min + frac * (y_max - y_min)
        y = _scale_y(val, height=height, pad=pad, y_min=y_min, y_max=y_max)
        lines.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        lines.append(
            f'<text x="{pad - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#6b7280">'
            f'{val:.2f}</text>'
        )
    return "\n  ".join(lines)


def _x_axis_labels(rows: list[dict[str, Any]], *, width: int, height: int, pad: int) -> str:
    labels = []
    min_iter = rows[0]["iteration"]
    max_iter = rows[-1]["iteration"]
    for row in rows:
        x = _scale_x(row["iteration"], min_iter=min_iter, max_iter=max_iter, width=width, pad=pad)
        labels.append(
            f'<text x="{x:.1f}" y="{height - pad + 22}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#374151">'
            f'{row["iteration"]}</text>'
        )
    return "\n  ".join(labels)


def _markers(rows: list[dict[str, Any]], *, width: int, height: int, pad: int, y_min: float, y_max: float) -> str:
    min_iter = rows[0]["iteration"]
    max_iter = rows[-1]["iteration"]
    markers = []
    labels = []
    for row in rows:
        x = _scale_x(row["iteration"], min_iter=min_iter, max_iter=max_iter, width=width, pad=pad)
        y = _scale_y(row["accuracy"], height=height, pad=pad, y_min=y_min, y_max=y_max)
        markers.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#ffffff" stroke="#0f766e" stroke-width="2.5"/>'
        )
        labels.append(
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="11" fill="#0f766e" font-weight="600">'
            f'{row["accuracy"]:.2%}</text>'
        )
    return "\n  ".join(markers), "\n  ".join(labels)


def render_svg(rows: list[dict[str, Any]]) -> str:
    width = 960
    height = 540
    pad = 70

    accuracies = [row["accuracy"] for row in rows]
    y_max = min(max(accuracies) * 1.12, 1.0)
    y_min = 0.0

    area_pts = _area_points(rows, width=width, height=height, pad=pad, y_min=y_min, y_max=y_max)
    line_pts = _line_points(rows, width=width, height=height, pad=pad, y_min=y_min, y_max=y_max)
    grid = _grid_lines(width=width, height=height, pad=pad, y_min=y_min, y_max=y_max)
    x_labels = _x_axis_labels(rows, width=width, height=height, pad=pad)
    markers, value_labels = _markers(rows, width=width, height=height, pad=pad, y_min=y_min, y_max=y_max)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.0"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="#f9fafb" rx="8" ry="8"/>
  <rect x="10" y="10" width="{width - 20}" height="{height - 20}" fill="#ffffff" rx="6" ry="6" stroke="#e5e7eb" stroke-width="1"/>

  <!-- Title -->
  <text x="{width / 2:.1f}" y="42" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="20" fill="#111827" font-weight="600">
    Evaluation Accuracy Curve
  </text>

  <!-- Grid + Y labels -->
  {grid}

  <!-- Axes -->
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#374151" stroke-width="2" stroke-linecap="round"/>

  <!-- Axis labels -->
  <text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#4b5563" font-weight="500">
    Iteration
  </text>
  <text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.1f})" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#4b5563" font-weight="500">
    Accuracy
  </text>

  <!-- Area under curve -->
  <polygon points="{area_pts}" fill="url(#areaGrad)"/>

  <!-- Main curve line with glow -->
  <polyline points="{line_pts}" fill="none" stroke="#14b8a6" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>
  <polyline points="{line_pts}" fill="none" stroke="#0f766e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- X-axis iteration labels -->
  {x_labels}

  <!-- Markers -->
  {markers}

  <!-- Value labels -->
  {value_labels}

  <!-- Legend -->
  <rect x="{width - pad - 140}" y="{pad + 10}" width="120" height="34" fill="#ffffff" stroke="#d1d5db" stroke-width="1" rx="4" ry="4"/>
  <line x1="{width - pad - 126}" y1="{pad + 27}" x2="{width - pad - 106}" y2="{pad + 27}" stroke="#0f766e" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="{width - pad - 116:.1f}" cy="{pad + 27:.1f}" r="3.5" fill="#ffffff" stroke="#0f766e" stroke-width="2"/>
  <text x="{width - pad - 98}" y="{pad + 31}" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="12" fill="#374151">Eval accuracy</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.eval_metrics)
    output_path = Path(args.output) if args.output else metrics_path.with_suffix(".svg")
    rows = load_metrics(metrics_path)
    output_path.write_text(render_svg(rows), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
