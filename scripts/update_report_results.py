from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


START = "| Agent | Horizon | Success Rate | Mean Return | Mean Length |"
END = "World model training curves should be inserted"


def build_table(results: pd.DataFrame) -> str:
    lines = [
        START,
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['agent']} | {int(row['horizon'])} | "
            f"{row['success_rate']:.3f} | {row['mean_return']:.3f} | {row['mean_length']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/paper.md")
    parser.add_argument("--results", default="runs/eval_results.csv")
    args = parser.parse_args()

    report_path = Path(args.report)
    results = pd.read_csv(args.results)
    text = report_path.read_text(encoding="utf-8")

    start_idx = text.index(START)
    end_idx = text.index(END, start_idx)
    new_text = text[:start_idx] + build_table(results) + "\n\n" + text[end_idx:]
    report_path.write_text(new_text, encoding="utf-8")
    print(f"updated {report_path} from {args.results}")


if __name__ == "__main__":
    main()
