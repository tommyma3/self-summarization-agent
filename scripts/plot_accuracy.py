from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/eval accuracy history as an SVG.")
    parser.add_argument(
        "accuracy_history",
        help="Path to artifacts/train/<experiment>/accuracy_history.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path. Defaults to accuracy_history.svg next to the input.",
    )
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "epoch": float(record["epoch"]),
                    "train_accuracy": float(record["train_accuracy"]),
                    "eval_accuracy": float(record["eval_accuracy"]),
                }
            )
    if not rows:
        raise ValueError(f"No accuracy records found in {path}")
    return rows


def _points(rows: list[dict[str, float]], key: str, *, width: int, height: int, pad: int) -> str:
    return " ".join(_xy_points(rows, key, width=width, height=height, pad=pad))


def _xy_points(rows: list[dict[str, float]], key: str, *, width: int, height: int, pad: int) -> list[str]:
    epochs = [row["epoch"] for row in rows]
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    epoch_span = max(max_epoch - min_epoch, 1.0)
    points = []
    for row in rows:
        x = pad + ((row["epoch"] - min_epoch) / epoch_span) * (width - 2 * pad)
        y = height - pad - row[key] * (height - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")
    return points


def _circles(rows: list[dict[str, float]], key: str, color: str, *, width: int, height: int, pad: int) -> str:
    circles = []
    for point in _xy_points(rows, key, width=width, height=height, pad=pad):
        x, y = point.split(",", 1)
        circles.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
    return "\n  ".join(circles)


def render_svg(rows: list[dict[str, float]]) -> str:
    width = 900
    height = 520
    pad = 64
    train_points = _points(rows, "train_accuracy", width=width, height=height, pad=pad)
    eval_points = _points(rows, "eval_accuracy", width=width, height=height, pad=pad)
    train_circles = _circles(rows, "train_accuracy", "#2563eb", width=width, height=height, pad=pad)
    eval_circles = _circles(rows, "eval_accuracy", "#dc2626", width=width, height=height, pad=pad)
    first_epoch = int(rows[0]["epoch"])
    last_epoch = int(rows[-1]["epoch"])

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#1f2937" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#1f2937" stroke-width="1.5"/>
  <line x1="{pad}" y1="{pad}" x2="{width - pad}" y2="{pad}" stroke="#e5e7eb" stroke-width="1"/>
  <line x1="{pad}" y1="{(height / 2):.1f}" x2="{width - pad}" y2="{(height / 2):.1f}" stroke="#e5e7eb" stroke-width="1"/>
  <text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#111827">Training Accuracy History</text>
  <text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#374151">Epoch</text>
  <text x="22" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {height / 2:.1f})" font-family="Arial, sans-serif" font-size="14" fill="#374151">Accuracy</text>
  <text x="{pad - 12}" y="{pad + 5}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">1.0</text>
  <text x="{pad - 12}" y="{height / 2 + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">0.5</text>
  <text x="{pad - 12}" y="{height - pad + 5}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">0.0</text>
  <text x="{pad}" y="{height - pad + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{first_epoch}</text>
  <text x="{width - pad}" y="{height - pad + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{last_epoch}</text>
  <polyline points="{train_points}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <polyline points="{eval_points}" fill="none" stroke="#dc2626" stroke-width="3"/>
  {train_circles}
  {eval_circles}
  <rect x="{width - 220}" y="64" width="156" height="58" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{width - 202}" y1="84" x2="{width - 166}" y2="84" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 154}" y="89" font-family="Arial, sans-serif" font-size="13" fill="#111827">Training set</text>
  <line x1="{width - 202}" y1="108" x2="{width - 166}" y2="108" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 154}" y="113" font-family="Arial, sans-serif" font-size="13" fill="#111827">Evaluation set</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    history_path = Path(args.accuracy_history)
    output_path = Path(args.output) if args.output else history_path.with_suffix(".svg")
    output_path.write_text(render_svg(load_history(history_path)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
