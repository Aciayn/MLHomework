from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_svg(path: str, body: str, width: int = 900, height: int = 520) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.bar{fill:#4c78a8}.line{fill:none;stroke:#f58518;stroke-width:3}.line2{fill:none;stroke:#4c78a8;stroke-width:3}</style>',
                body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    print(f"saved {output}")


def plot_success() -> None:
    rows = read_csv("runs/server_eval_results.csv")
    labels = [row["agent"] for row in rows]
    values = [float(row["success_rate"]) for row in rows]
    width, height = 900, 520
    left, top, chart_w, chart_h = 90, 80, 760, 320
    bar_w = chart_w / len(values) * 0.62
    gap = chart_w / len(values)

    parts = [
        '<text x="450" y="38" text-anchor="middle" font-size="24">Server Evaluation: Success Rate</text>',
        f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>',
    ]
    for i in range(6):
        y = top + chart_h - i * chart_h / 5
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="13">{i / 5:.1f}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        h = value * chart_h
        y = top + chart_h - h
        parts.append(f'<rect class="bar" x="{x}" y="{y}" width="{bar_w}" height="{h}"/>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{y - 8}" text-anchor="middle" font-size="13">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{top + chart_h + 28}" text-anchor="middle" font-size="12">{label}</text>')
    parts.append('<text x="26" y="240" transform="rotate(-90 26 240)" text-anchor="middle" font-size="15">Success Rate</text>')
    write_svg("reports/figures/server_success_rate.svg", "\n".join(parts), width, height)


def plot_loss() -> None:
    rows = read_csv("runs/server_world_model.history.csv")
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    val = [float(row["val_loss"]) for row in rows]
    width, height = 900, 520
    left, top, chart_w, chart_h = 90, 80, 760, 320
    max_loss = max(train + val) * 1.05
    min_loss = min(train + val) * 0.9

    def point(epoch: int, value: float) -> tuple[float, float]:
        x = left + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * chart_w
        y = top + chart_h - (value - min_loss) / (max_loss - min_loss) * chart_h
        return x, y

    train_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(e, v) for e, v in zip(epochs, train)))
    val_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(e, v) for e, v in zip(epochs, val)))
    parts = [
        '<text x="450" y="38" text-anchor="middle" font-size="24">World Model Training Loss</text>',
        f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>',
        f'<polyline class="line2" points="{train_points}"/>',
        f'<polyline class="line" points="{val_points}"/>',
        '<rect x="675" y="75" width="14" height="14" fill="#4c78a8"/><text x="696" y="88" font-size="14">train</text>',
        '<rect x="675" y="100" width="14" height="14" fill="#f58518"/><text x="696" y="113" font-size="14">validation</text>',
        f'<text x="{left}" y="{top + chart_h + 35}" font-size="14">epoch {epochs[0]}</text>',
        f'<text x="{left + chart_w}" y="{top + chart_h + 35}" text-anchor="end" font-size="14">epoch {epochs[-1]}</text>',
        f'<text x="{left - 12}" y="{top + 5}" text-anchor="end" font-size="13">{max_loss:.3f}</text>',
        f'<text x="{left - 12}" y="{top + chart_h}" text-anchor="end" font-size="13">{min_loss:.3f}</text>',
        '<text x="450" y="485" text-anchor="middle" font-size="15">Epoch</text>',
        '<text x="26" y="240" transform="rotate(-90 26 240)" text-anchor="middle" font-size="15">Loss</text>',
    ]
    write_svg("reports/figures/server_world_model_loss.svg", "\n".join(parts), width, height)


def plot_multiseed() -> None:
    path = Path("runs/multiseed_summary.csv")
    if not path.exists():
        return

    rows = read_csv(str(path))
    labels = [row["agent"] for row in rows]
    values = [float(row["success_rate_mean"]) for row in rows]
    errors = [float(row["success_rate_std"]) for row in rows]
    width, height = 900, 520
    left, top, chart_w, chart_h = 90, 80, 760, 320
    bar_w = chart_w / len(values) * 0.62
    gap = chart_w / len(values)

    parts = [
        '<text x="450" y="38" text-anchor="middle" font-size="24">Multi-Seed Success Rate</text>',
        f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>',
    ]
    for i in range(6):
        y = top + chart_h - i * chart_h / 5
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="13">{i / 5:.1f}</text>')
    for idx, (label, value, error) in enumerate(zip(labels, values, errors)):
        x = left + idx * gap + (gap - bar_w) / 2
        h = value * chart_h
        y = top + chart_h - h
        err_h = error * chart_h
        cx = x + bar_w / 2
        parts.append(f'<rect class="bar" x="{x}" y="{y}" width="{bar_w}" height="{h}"/>')
        parts.append(f'<line class="axis" x1="{cx}" y1="{max(top, y - err_h)}" x2="{cx}" y2="{min(top + chart_h, y + err_h)}"/>')
        parts.append(f'<line class="axis" x1="{cx - 8}" y1="{max(top, y - err_h)}" x2="{cx + 8}" y2="{max(top, y - err_h)}"/>')
        parts.append(f'<line class="axis" x1="{cx - 8}" y1="{min(top + chart_h, y + err_h)}" x2="{cx + 8}" y2="{min(top + chart_h, y + err_h)}"/>')
        parts.append(f'<text x="{cx}" y="{y - err_h - 8}" text-anchor="middle" font-size="13">{value:.2f}</text>')
        parts.append(f'<text x="{cx}" y="{top + chart_h + 28}" text-anchor="middle" font-size="12">{label}</text>')
    parts.append('<text x="26" y="240" transform="rotate(-90 26 240)" text-anchor="middle" font-size="15">Success Rate</text>')
    write_svg("reports/figures/multiseed_success_rate.svg", "\n".join(parts), width, height)


def main() -> None:
    plot_success()
    plot_loss()
    plot_multiseed()


if __name__ == "__main__":
    main()
