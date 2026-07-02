from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize training sequence lengths from a rollout JSONL file."
    )
    parser.add_argument(
        "input",
        help="Path to a rollout JSONL file (e.g. iteration-00001.judged.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path. Defaults to <input_stem>_seq_lengths.svg alongside the input.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path to dump per-sample lengths.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Cap the histogram x-axis at this token count (default: auto from data).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=None,
        help="Histogram bin width in tokens (default: auto via sqrt rule).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_lengths(jsonl_path: Path) -> list[dict[str, int]]:
    """Parse a rollout JSONL and return per-sample token length records.

    Each returned dict has keys: *total*, *prompt*, *completion*.
    """
    records: list[dict[str, int]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping line {line_num} — invalid JSON: {exc}")
                continue
            turn_records = row.get("turn_records")
            if not isinstance(turn_records, list):
                continue
            for turn in turn_records:
                cache = turn.get("training_cache")
                if not isinstance(cache, dict):
                    continue
                input_ids = cache.get("input_ids")
                completion_mask = cache.get("completion_mask")
                if not isinstance(input_ids, list) or not isinstance(completion_mask, list):
                    continue
                total = len(input_ids)
                prompt = sum(1 for m in completion_mask if not m)
                completion = total - prompt
                records.append({"total": total, "prompt": prompt, "completion": completion})
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        """Linear-interpolation percentile (0–100)."""
        if n == 0:
            return 0.0
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_vals[int(k)])
        d0 = sorted_vals[f] * (c - k)
        d1 = sorted_vals[c] * (k - f)
        return float(d0 + d1)

    return {
        "count": n,
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "stdev": statistics.stdev(sorted_vals) if n >= 2 else 0.0,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def _fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"  count={int(stats['count']):,}"
        f"  mean={stats['mean']:,.1f}"
        f"  median={stats['median']:,.0f}"
        f"  stdev={stats['stdev']:,.1f}"
        f"  min={stats['min']:,.0f}"
        f"  max={stats['max']:,.0f}"
        f"  p90={stats['p90']:,.0f}"
        f"  p95={stats['p95']:,.0f}"
        f"  p99={stats['p99']:,.0f}"
    )


def print_stats(records: list[dict[str, int]]) -> None:
    totals = [r["total"] for r in records]
    prompts = [r["prompt"] for r in records]
    completions = [r["completion"] for r in records]

    print(f"\nSequence Length Analysis — {len(records):,} trainable samples\n")
    print(f"{'':>14}  {'Total':>10}  {'Prompt':>10}  {'Completion':>10}")
    print("-" * 52)

    total_stats = compute_stats(totals)
    prompt_stats = compute_stats(prompts)
    completion_stats = compute_stats(completions)

    for label in ["count", "mean", "median", "stdev", "min", "max", "p90", "p95", "p99"]:
        t_val = total_stats.get(label, 0)
        p_val = prompt_stats.get(label, 0)
        c_val = completion_stats.get(label, 0)
        if label == "count":
            fmt = "{:>14}  {:>10,.0f}  {:>10,.0f}  {:>10,.0f}"
        else:
            fmt = "{:>14}  {:>10,.1f}  {:>10,.1f}  {:>10,.1f}"
        print(fmt.format(label, t_val, p_val, c_val))


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def build_histogram(
    values: list[int], *, bin_size: int | None = None, max_length: int | None = None
) -> tuple[list[int], int, int]:
    """Return (bin_counts, bin_start, bin_size).

    Bins are aligned to multiples of *bin_size*.  If *max_length* is given,
    the histogram is capped at that value (values beyond it are counted in
    the final bin).
    """
    if not values:
        return [], 0, 1

    data_max = max(values)
    if max_length is not None:
        data_max = min(data_max, max_length)

    if bin_size is None:
        # sqrt rule for number of bins
        n_bins = max(1, int(math.sqrt(len(values))))
        bin_size = max(1, math.ceil(data_max / n_bins))
        # round to a "nice" number
        nice = _nice_round(bin_size)
        bin_size = max(1, nice)

    num_bins = math.ceil(data_max / bin_size) if data_max > 0 else 1
    counts = [0] * num_bins

    for v in values:
        idx = min(v // bin_size, num_bins - 1)
        counts[idx] += 1

    return counts, 0, bin_size


def _nice_round(n: int) -> int:
    """Round *n* up to the nearest 'nice' number (1, 2, 5, 10, 20, 50, …)."""
    if n <= 1:
        return 1
    magnitude = 10 ** math.floor(math.log10(n))
    residual = n / magnitude
    for candidate in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if candidate >= residual:
            return int(candidate * magnitude)
    return n


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------


def _scale_x(value: float, *, x_min: float, x_max: float, width: int, pad: int) -> float:
    span = max(x_max - x_min, 1)
    return pad + ((value - x_min) / span) * (width - 2 * pad)


def _scale_y(value: float, *, y_min: float, y_max: float, height: int, pad: int) -> float:
    span = max(y_max - y_min, 1)
    return height - pad - ((value - y_min) / span) * (height - 2 * pad)


def _render_histogram_panel(
    counts: list[int],
    bin_size: int,
    bin_start: int,
    *,
    label: str,
    color: str,
    width: int,
    height: int,
    pad: int,
    x_offset: float = 0,
    y_offset: float = 0,
    x_label: str = "Tokens",
    title: str = "",
    stats_text: str = "",
) -> str:
    if not counts:
        return ""

    y_max = max(counts) * 1.12 if max(counts) > 0 else 1
    x_min = bin_start
    x_max = bin_start + len(counts) * bin_size

    bars: list[str] = []
    bar_width = ((width - 2 * pad) / len(counts)) * 0.85 if len(counts) > 0 else 0

    for i, count in enumerate(counts):
        bin_center = x_offset + bin_start + (i + 0.5) * bin_size
        bar_x = _scale_x(bin_start + i * bin_size, x_min=x_min, x_max=x_max, width=width, pad=pad)
        bar_h = max(0, _scale_y(0, y_min=0, y_max=y_max, height=height, pad=pad) - _scale_y(count, y_min=0, y_max=y_max, height=height, pad=pad))
        bar_y = _scale_y(count, y_min=0, y_max=y_max, height=height, pad=pad)

        bars.append(
            f'<rect x="{bar_x + (width - 2 * pad) / len(counts) * 0.075:.1f}" '
            f'y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" fill-opacity="0.75" stroke="{color}" stroke-width="0.5"/>'
        )

    # Y-axis grid lines + labels
    grid_lines: list[str] = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = frac * y_max
        y = _scale_y(val, y_min=0, y_max=y_max, height=height, pad=pad) + y_offset
        grid_lines.append(
            f'<line x1="{x_offset + pad}" y1="{y:.1f}" x2="{x_offset + width - pad}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        grid_lines.append(
            f'<text x="{x_offset + pad - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="11" fill="#6b7280">'
            f'{val:.0f}</text>'
        )

    # X-axis labels (show a few tick marks)
    x_ticks: list[str] = []
    num_ticks = min(8, len(counts))
    step = max(1, len(counts) // num_ticks)
    for i in range(0, len(counts), step):
        tick_val = bin_start + i * bin_size
        x = _scale_x(tick_val, x_min=x_min, x_max=x_max, width=width, pad=pad) + x_offset
        x_ticks.append(
            f'<text x="{x:.1f}" y="{y_offset + height - pad + 20}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="11" fill="#4b5563">'
            f'{tick_val}</text>'
        )

    # Panel border
    border = (
        f'<rect x="{x_offset + pad - 2}" y="{y_offset + pad - 2}" '
        f'width="{width - 2 * pad + 4}" height="{height - 2 * pad + 4}" '
        f'fill="none" stroke="#d1d5db" stroke-width="1" rx="4"/>'
    )

    parts: list[str] = [
        border,
        f'<line x1="{x_offset + pad}" y1="{y_offset + height - pad}" x2="{x_offset + width - pad}" y2="{y_offset + height - pad}" stroke="#374151" stroke-width="1.5"/>',
        f'<line x1="{x_offset + pad}" y1="{y_offset + pad}" x2="{x_offset + pad}" y2="{y_offset + height - pad}" stroke="#374151" stroke-width="1.5"/>',
        *grid_lines,
        *bars,
        *x_ticks,
        # X-axis label
        f'<text x="{x_offset + width / 2:.1f}" y="{y_offset + height - 6}" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="13" fill="#4b5563" font-weight="500">{x_label}</text>',
        # Y-axis label
        f'<text x="{x_offset + 14}" y="{y_offset + height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {x_offset + 14} {y_offset + height / 2:.1f})" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="13" fill="#4b5563" font-weight="500">Samples</text>',
    ]

    if title:
        parts.append(
            f'<text x="{x_offset + width / 2:.1f}" y="{y_offset + 22}" text-anchor="middle" '
            f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="14" fill="#111827" font-weight="600">{title}</text>'
        )

    if stats_text:
        for i, line in enumerate(stats_text.strip().split("\n")):
            parts.append(
                f'<text x="{x_offset + width - pad - 4}" y="{y_offset + pad + 16 + i * 16}" text-anchor="end" '
                f'font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="10" fill="#6b7280">{line}</text>'
            )

    return "\n  ".join(parts)


def render_svg(
    records: list[dict[str, int]],
    *,
    bin_size: int | None = None,
    max_length: int | None = None,
) -> str:
    totals = [r["total"] for r in records]
    prompts = [r["prompt"] for r in records]
    completions = [r["completion"] for r in records]

    total_counts, total_bin_start, total_bin_size = build_histogram(
        totals, bin_size=bin_size, max_length=max_length
    )
    prompt_counts, prompt_bin_start, _prompt_bs = build_histogram(
        prompts, bin_size=bin_size or total_bin_size, max_length=max_length
    )
    completion_counts, comp_bin_start, _comp_bs = build_histogram(
        completions, bin_size=bin_size or total_bin_size, max_length=max_length
    )

    # Use the same bin_size across all panels
    shared_bin_size = total_bin_size
    if bin_size is None:
        shared_bin_size = max(total_bin_size, _prompt_bs, _comp_bs)

    # Rebuild with shared bin size
    total_counts, total_bin_start, total_bin_size = build_histogram(
        totals, bin_size=shared_bin_size, max_length=max_length
    )
    prompt_counts, prompt_bin_start, _ = build_histogram(
        prompts, bin_size=shared_bin_size, max_length=max_length
    )
    completion_counts, comp_bin_start, _ = build_histogram(
        completions, bin_size=shared_bin_size, max_length=max_length
    )

    total_stats = compute_stats(totals)
    prompt_stats = compute_stats(prompts)
    comp_stats = compute_stats(completions)

    width = 1000
    height = 720
    pad = 68
    panel_height = (height - 3 * pad) // 2

    total_stats_text = (
        f"Total — mean={total_stats.get('mean', 0):.0f}  "
        f"median={total_stats.get('median', 0):.0f}  "
        f"max={total_stats.get('max', 0):.0f}  "
        f"p95={total_stats.get('p95', 0):.0f}"
    )
    sub_stats_text = (
        f"Prompt — mean={prompt_stats.get('mean', 0):.0f}  "
        f"p95={prompt_stats.get('p95', 0):.0f}  "
        f"max={prompt_stats.get('max', 0):.0f}\n"
        f"Completion — mean={comp_stats.get('mean', 0):.0f}  "
        f"p95={comp_stats.get('p95', 0):.0f}  "
        f"max={comp_stats.get('max', 0):.0f}"
    )

    # Top panel — total lengths
    top_panel = _render_histogram_panel(
        total_counts,
        total_bin_size,
        total_bin_start,
        label="Total",
        color="#2563eb",
        width=width,
        height=panel_height,
        pad=pad,
        y_offset=pad,
        title=f"Total Sequence Lengths  ({len(records):,} samples,  bin={total_bin_size} tokens)",
        stats_text=total_stats_text,
    )

    # Bottom panel — prompt vs completion overlaid
    bottom_panel = _render_histogram_panel(
        prompt_counts,
        shared_bin_size,
        prompt_bin_start,
        label="Prompt",
        color="#dc2626",
        width=width,
        height=panel_height,
        pad=pad,
        x_offset=0,
        y_offset=pad * 2 + panel_height,
        title=f"Prompt (red)  vs  Completion (blue) Lengths  (bin={shared_bin_size} tokens)",
        stats_text=sub_stats_text,
    )

    # Overlay completion bars on the bottom panel
    comp_bars: list[str] = []
    if completion_counts and prompt_counts:
        y_max_bottom = max(max(prompt_counts), max(completion_counts)) * 1.12
        comp_bar_width = ((width - 2 * pad) / len(completion_counts)) * 0.4 if len(completion_counts) > 0 else 0
        prompt_bar_width = ((width - 2 * pad) / len(prompt_counts)) * 0.4 if len(prompt_counts) > 0 else 0
        offset = ((width - 2 * pad) / len(completion_counts)) * 0.5 if len(completion_counts) > 0 else 0

        for i, count in enumerate(completion_counts):
            bar_x = _scale_x(comp_bin_start + i * shared_bin_size, x_min=prompt_bin_start, x_max=prompt_bin_start + len(prompt_counts) * shared_bin_size, width=width, pad=pad)
            bar_h = max(0, _scale_y(0, y_min=0, y_max=y_max_bottom, height=panel_height, pad=pad) - _scale_y(count, y_min=0, y_max=y_max_bottom, height=panel_height, pad=pad))
            bar_y = _scale_y(count, y_min=0, y_max=y_max_bottom, height=panel_height, pad=pad) + pad * 2 + panel_height

            comp_bars.append(
                f'<rect x="{bar_x + offset:.1f}" y="{bar_y:.1f}" '
                f'width="{comp_bar_width:.1f}" height="{bar_h:.1f}" '
                f'fill="#2563eb" fill-opacity="0.6" stroke="#2563eb" stroke-width="0.5"/>'
            )

    # Legend for bottom panel
    legend_y = pad * 2 + panel_height + pad - 10
    legend = (
        f'<rect x="{width - pad - 170}" y="{legend_y}" width="150" height="34" fill="#ffffff" fill-opacity="0.9" stroke="#d1d5db" stroke-width="1" rx="4"/>'
        f'<rect x="{width - pad - 158}" y="{legend_y + 7}" width="14" height="8" fill="#dc2626" fill-opacity="0.75"/>'
        f'<text x="{width - pad - 138}" y="{legend_y + 15}" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="11" fill="#374151">Prompt</text>'
        f'<rect x="{width - pad - 158}" y="{legend_y + 21}" width="14" height="8" fill="#2563eb" fill-opacity="0.6"/>'
        f'<text x="{width - pad - 138}" y="{legend_y + 29}" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="11" fill="#374151">Completion</text>'
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- Background -->
  <rect width="100%" height="100%" fill="#f9fafb"/>
  <rect x="8" y="8" width="{width - 16}" height="{height - 16}" fill="#ffffff" rx="6" ry="6" stroke="#e5e7eb" stroke-width="1"/>

  <!-- Title -->
  <text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="ui-sans-serif, system-ui, Arial, sans-serif" font-size="22" fill="#111827" font-weight="700">
    Training Sequence Length Distribution
  </text>

  <!-- Top panel — total lengths -->
  {top_panel}

  <!-- Bottom panel — prompt vs completion -->
  {bottom_panel}

  <!-- Completion overlay bars -->
  {"\n  ".join(comp_bars)}

  <!-- Legend -->
  {legend}
</svg>
"""


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_csv(records: list[dict[str, int]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["total", "prompt", "completion"])
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV written to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = extract_lengths(input_path)
    if not records:
        print("No trainable samples found in the input file.")
        return

    # Text summary
    print_stats(records)

    # SVG
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_seq_lengths.svg"
    svg = render_svg(records, bin_size=args.bin_size, max_length=args.max_length)
    output_path.write_text(svg, encoding="utf-8")
    print(f"\nSVG written to {output_path}")

    # Optional CSV
    if args.csv:
        write_csv(records, Path(args.csv))


if __name__ == "__main__":
    main()
