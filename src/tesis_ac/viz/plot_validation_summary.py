import csv
import os
from typing import List, Dict
import matplotlib.pyplot as plt

BASE = "/Users/rod/Projects/MSC/Tesis/CityModelling"
CSV_DEFAULT = os.path.join(BASE, "reports", "validation_summary.csv")
CSV_INVERT = os.path.join(BASE, "reports", "validation_summary_invert.csv")
CSV_PATH = CSV_INVERT if os.path.exists(CSV_INVERT) else CSV_DEFAULT
OUT_DIR = os.path.join(BASE, "figures")
OUT_PATH = os.path.join(OUT_DIR, "validation_metrics.png")


def load_summary(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Sort by period start year
    rows.sort(key=lambda r: int(r["period"].split("→")[0]))
    return rows


def plot_metrics(rows: List[Dict[str, str]], out_path: str) -> None:
    periods = [r["period"] for r in rows]
    def col(name):
        return [float(r[name]) for r in rows]

    metrics = [
        ("accuracy", col("accuracy")),
        ("kappa", col("kappa")),
        ("IoU", col("iou")),
        ("FoM", col("fom")),
        ("quantity_disagreement", col("quantity_disagreement")),
        ("allocation_disagreement", col("allocation_disagreement")),
        ("overall_disagreement", col("overall_disagreement")),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, values) in enumerate(metrics):
        ax = axes[i]
        ax.plot(periods, values, marker="o")
        ax.set_title(name)
        ax.set_xticks(periods)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    # Hide the last empty subplot if metrics < 8
    if len(metrics) < len(axes):
        axes[-1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    rows = load_summary(CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_metrics(rows, OUT_PATH)
