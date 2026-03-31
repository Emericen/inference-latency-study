"""Generate lightweight SVG plots for the README from summary CSV files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES_DIR = ROOT / "results" / "summaries"
PLOTS_DIR = ROOT / "assets" / "plots"

BACKGROUND = "#fffdfa"
GRID = "#d7d1cb"
AXIS = "#3f3a37"
TEXT = "#2e2a28"
SUBTLE = "#6f6762"
PALETTE = [
    "#98dfc2",
    "#79d1b0",
    "#62bdb3",
    "#5099bb",
    "#456eb4",
    "#483f9a",
    "#3a2e66",
    "#22192e",
]


@dataclass
class Series:
    label: str
    x: list[float]
    y: list[float]
    color: str


def _svg_line_plot(
    *,
    x_label: str,
    y_label: str,
    series: list[Series],
    output_path: Path,
) -> None:
    width = 1040
    height = 560
    left = 90
    right = 220
    top = 32
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [value for s in series for value in s.x]
    ys = [value for s in series for value in s.y]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    y_floor = 0.0
    y_ceiling = max(y_max * 1.08, 0.1)
    if y_ceiling == y_floor:
        y_ceiling = y_floor + 1.0

    def x_px(value: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + ((value - x_min) / (x_max - x_min)) * plot_w

    def y_px(value: float) -> float:
        return top + plot_h - ((value - y_floor) / (y_ceiling - y_floor)) * plot_h

    x_ticks = [1, 10, 25, 50, 75, 100]
    x_ticks = [tick for tick in x_ticks if x_min <= tick <= x_max]

    tick_count = 6
    raw_step = (y_ceiling - y_floor) / tick_count
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 0.1
    for candidate in [1, 2, 2.5, 5, 10]:
        step = candidate * magnitude
        if step >= raw_step:
            break
    y_ticks = []
    value = y_floor
    while value <= y_ceiling + step * 0.5:
        y_ticks.append(round(value, 3))
        value += step

    grid_lines = []
    for tick in x_ticks:
        px = x_px(tick)
        grid_lines.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
    for tick in y_ticks:
        py = y_px(tick)
        grid_lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )

    series_markup = []
    for s in series:
        points = " ".join(f"{x_px(x):.1f},{y_px(y):.1f}" for x, y in zip(s.x, s.y))
        series_markup.append(
            f'<polyline fill="none" stroke="{s.color}" stroke-width="4" '
            f'stroke-linecap="round" stroke-linejoin="round" points="{points}"/>'
        )
        for x, y in zip(s.x, s.y):
            series_markup.append(
                f'<circle cx="{x_px(x):.1f}" cy="{y_px(y):.1f}" r="4.5" '
                f'fill="{s.color}" fill-opacity="0.95"/>'
            )

    legend_x = left + plot_w + 26
    legend_y = top + 30
    legend = [
        f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="180" height="{len(series) * 34 + 20}" '
        f'rx="14" fill="{BACKGROUND}" stroke="{GRID}" stroke-width="1.2"/>'
    ]
    for i, s in enumerate(series):
        y = legend_y + i * 34
        legend.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" '
            f'stroke="{s.color}" stroke-width="4" stroke-linecap="round"/>'
        )
        legend.append(
            f'<circle cx="{legend_x + 17}" cy="{y}" r="4.5" fill="{s.color}"/>'
        )
        legend.append(
            f'<text x="{legend_x + 48}" y="{y + 7}" font-size="21" '
            f'font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{s.label}</text>'
        )

    x_tick_labels = []
    for tick in x_ticks:
        px = x_px(tick)
        x_tick_labels.append(
            f'<text x="{px:.1f}" y="{top + plot_h + 32}" text-anchor="middle" '
            f'font-size="18" font-family="Helvetica, Arial, sans-serif" fill="{SUBTLE}">{tick}</text>'
        )

    y_tick_labels = []
    for tick in y_ticks:
        py = y_px(tick)
        label = f"{tick:.1f}" if tick >= 1 else f"{tick:.2f}"
        y_tick_labels.append(
            f'<text x="{left - 14}" y="{py + 6:.1f}" text-anchor="end" '
            f'font-size="18" font-family="Helvetica, Arial, sans-serif" fill="{SUBTLE}">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{BACKGROUND}"/>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{AXIS}" stroke-width="2"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{AXIS}" stroke-width="2"/>
  {''.join(series_markup)}
  {''.join(legend)}
  {''.join(x_tick_labels)}
  {''.join(y_tick_labels)}
  <text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="20" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{x_label}</text>
  <g transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)">
    <text text-anchor="middle" font-size="20" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{y_label}</text>
  </g>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(SUMMARIES_DIR / path)


def _make_series(df: pd.DataFrame, *, label: str, color: str) -> Series:
    return Series(
        label=label,
        x=df["context_size"].tolist(),
        y=df["ttft_p50_s"].tolist(),
        color=color,
    )


def generate_plots() -> list[Path]:
    outputs: list[Path] = []

    dense_scale = [
        ("2B", _load("dense_scale_qwen3_vl_2b_local.csv"), PALETTE[1]),
        ("4B", _load("dense_scale_qwen3_vl_4b_local.csv"), PALETTE[3]),
        ("8B", _load("dense_scale_qwen3_vl_8b_local.csv"), PALETTE[5]),
    ]
    dense_scale_path = PLOTS_DIR / "dense_scale_local_ttft.svg"
    _svg_line_plot(
        x_label="Context Size",
        y_label="TTFT (seconds)",
        series=[_make_series(df, label=label, color=color) for label, df, color in dense_scale],
        output_path=dense_scale_path,
    )
    outputs.append(dense_scale_path)

    moe_vs_dense = [
        ("30B-A3B MoE", _load("dense_vs_moe_qwen3_vl_30b_a3b_moe_local.csv"), PALETTE[4]),
        ("32B Dense", _load("dense_vs_moe_qwen3_vl_32b_dense_local.csv"), PALETTE[7]),
    ]
    moe_path = PLOTS_DIR / "dense_vs_moe_local_ttft.svg"
    _svg_line_plot(
        x_label="Context Size",
        y_label="TTFT (seconds)",
        series=[_make_series(df, label=label, color=color) for label, df, color in moe_vs_dense],
        output_path=moe_path,
    )
    outputs.append(moe_path)

    full_history = [
        ("local", _load("screenshot_history_full_local.csv"), PALETTE[1]),
        ("us-ca-2", _load("screenshot_history_full_us_ca_2.csv"), PALETTE[4]),
        ("us-mo-1", _load("screenshot_history_full_us_mo_1.csv"), PALETTE[7]),
    ]
    full_path = PLOTS_DIR / "full_screenshot_history_by_region_ttft.svg"
    _svg_line_plot(
        x_label="Context Size",
        y_label="TTFT (seconds)",
        series=[_make_series(df, label=label, color=color) for label, df, color in full_history],
        output_path=full_path,
    )
    outputs.append(full_path)

    omitted_history = [
        ("local", _load("screenshot_history_omit_past_local.csv"), PALETTE[1]),
        ("us-ca-2", _load("screenshot_history_omit_past_us_ca_2.csv"), PALETTE[4]),
        ("us-mo-1", _load("screenshot_history_omit_past_us_mo_1.csv"), PALETTE[7]),
    ]
    omit_path = PLOTS_DIR / "omit_past_screenshot_history_by_region_ttft.svg"
    _svg_line_plot(
        x_label="Context Size",
        y_label="TTFT (seconds)",
        series=[_make_series(df, label=label, color=color) for label, df, color in omitted_history],
        output_path=omit_path,
    )
    outputs.append(omit_path)

    return outputs


def main() -> None:
    outputs = generate_plots()
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
