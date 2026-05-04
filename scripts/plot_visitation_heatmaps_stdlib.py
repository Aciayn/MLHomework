from __future__ import annotations

import csv
from pathlib import Path


AGENT_TITLES = {
    "random": "Random",
    "mpc": "WM-MPC-H7",
    "ppo": "PPO",
}


def read_counts(path: str) -> dict[str, dict[tuple[int, int], int]]:
    counts: dict[str, dict[tuple[int, int], int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agent = row["agent"]
            counts.setdefault(agent, {})[(int(row["x"]), int(row["y"]))] = int(row["count"])
    return counts


def read_metrics(path: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics[row["agent"]] = {
                "success_rate": float(row["success_rate"]),
                "mean_length": float(row["mean_length"]),
            }
    return metrics


def color(value: int, vmax: int) -> str:
    if value <= 0 or vmax <= 0:
        return "#f7f7f7"
    ratio = min(1.0, value / vmax)
    # Simple white -> orange -> dark purple scale.
    r = int(247 - 150 * ratio)
    g = int(247 - 210 * ratio)
    b = int(247 - 115 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    counts = read_counts("runs/state_visitation_counts.csv")
    metrics = read_metrics("runs/state_visitation_metrics.csv")
    agents = ["random", "mpc", "ppo"]
    vmax = max(max(agent_counts.values()) for agent_counts in counts.values())
    cell = 42
    gap = 36
    top = 88
    left = 54
    panel_w = cell * 8
    width = left * 2 + panel_w * 3 + gap * 2
    height = 500

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.grid{stroke:#555;stroke-width:0.7}.wall{fill:#111}.goal{fill:#36c36b;stroke:#111;stroke-width:1}</style>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="24">State Visitation Heatmaps over 100 Episodes</text>',
    ]

    for panel_idx, agent in enumerate(agents):
        x0 = left + panel_idx * (panel_w + gap)
        title = AGENT_TITLES[agent]
        metric = metrics[agent]
        parts.append(f'<text x="{x0 + panel_w / 2}" y="66" text-anchor="middle" font-size="17">{title}: success={metric["success_rate"]:.2f}, length={metric["mean_length"]:.1f}</text>')
        for y in range(8):
            for x in range(8):
                count = counts[agent].get((x, y), 0)
                fill = "#111111" if x in (0, 7) or y in (0, 7) else color(count, vmax)
                if (x, y) == (6, 6):
                    fill = "#38d65a"
                parts.append(
                    f'<rect class="grid" x="{x0 + x * cell}" y="{top + y * cell}" '
                    f'width="{cell}" height="{cell}" fill="{fill}"/>'
                )
                if count > 0 and (x, y) != (6, 6):
                    font_size = 10 if count < 1000 else 9
                    parts.append(
                        f'<text x="{x0 + x * cell + cell / 2}" y="{top + y * cell + cell / 2 + 4}" '
                        f'text-anchor="middle" font-size="{font_size}">{count}</text>'
                    )
        parts.append(f'<text x="{x0 + 6 * cell + cell / 2}" y="{top + 6 * cell + cell / 2 + 5}" text-anchor="middle" font-size="11">Goal</text>')

    parts.extend(
        [
            f'<text x="{left}" y="455" font-size="13">Darker cells indicate more visits. Black cells are walls; green cell is the goal.</text>',
            "</svg>",
        ]
    )
    output = Path("reports/figures/state_visitation_heatmaps.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
