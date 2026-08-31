"""Paired comparison of two LiShift batch result files."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean, median
from typing import Dict, Iterable, Mapping

import numpy as np


METRIC_KEYS = ("SSI", "GCP", "SSD", "STR", "CSI", "COI", "LQI")


def _load_batch_result(path: str) -> Dict:
    """Load and validate a batch result JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    if not isinstance(result, dict) or not isinstance(result.get("characters"), dict):
        raise ValueError(f"Not a LiShift batch result file: {path}")
    return result


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _has_metrics(item: Mapping[str, object]) -> bool:
    return all(_is_number(item.get(metric)) for metric in METRIC_KEYS)


def compare_result_files(
    before_path: str,
    after_path: str,
    min_samples: int = 1,
) -> Dict:
    """Compare common characters using paired ``after - before`` differences.

    Args:
        before_path: Earlier/reference LiShift batch result JSON.
        after_path: Later/comparison LiShift batch result JSON.
        min_samples: Minimum image count required in both datasets.

    Returns:
        A JSON-serializable paired comparison report.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    before_result = _load_batch_result(before_path)
    after_result = _load_batch_result(after_path)
    before_chars = before_result["characters"]
    after_chars = after_result["characters"]

    common_chars = sorted(set(before_chars) & set(after_chars))
    compared: Dict[str, Dict] = {}
    excluded_by_min_samples = 0
    excluded_invalid_metrics = 0

    for char in common_chars:
        before = before_chars[char]
        after = after_chars[char]
        if not isinstance(before, dict) or not isinstance(after, dict):
            excluded_invalid_metrics += 1
            continue
        if not _has_metrics(before) or not _has_metrics(after):
            excluded_invalid_metrics += 1
            continue

        before_count = int(before.get("image_count", 0))
        after_count = int(after.get("image_count", 0))
        if before_count < min_samples or after_count < min_samples:
            excluded_by_min_samples += 1
            continue

        before_metrics = {key: float(before[key]) for key in METRIC_KEYS}
        after_metrics = {key: float(after[key]) for key in METRIC_KEYS}
        deltas = {
            key: after_metrics[key] - before_metrics[key]
            for key in METRIC_KEYS
        }
        compared[char] = {
            "before": {**before_metrics, "image_count": before_count},
            "after": {**after_metrics, "image_count": after_count},
            "delta": deltas,
        }

    metric_summary = {}
    for metric in METRIC_KEYS:
        before_values = [item["before"][metric] for item in compared.values()]
        after_values = [item["after"][metric] for item in compared.values()]
        deltas = [item["delta"][metric] for item in compared.values()]
        if not deltas:
            metric_summary[metric] = {
                "before_mean": None,
                "after_mean": None,
                "mean_delta": None,
                "median_delta": None,
                "after_higher_count": 0,
                "after_higher_rate": None,
            }
            continue

        after_higher_count = sum(delta > 0 for delta in deltas)
        metric_summary[metric] = {
            "before_mean": float(fmean(before_values)),
            "after_mean": float(fmean(after_values)),
            "mean_delta": float(fmean(deltas)),
            "median_delta": float(median(deltas)),
            "after_higher_count": after_higher_count,
            "after_higher_rate": after_higher_count / len(deltas),
        }

    return {
        "comparison": {
            "before_dataset": before_result.get("dataset_name", Path(before_path).stem),
            "after_dataset": after_result.get("dataset_name", Path(after_path).stem),
            "before_path": os.path.abspath(before_path),
            "after_path": os.path.abspath(after_path),
            "direction": "after - before",
            "min_samples": min_samples,
        },
        "summary": {
            "before_total_characters": len(before_chars),
            "after_total_characters": len(after_chars),
            "common_characters": len(common_chars),
            "compared_characters": len(compared),
            "excluded_by_min_samples": excluded_by_min_samples,
            "excluded_invalid_metrics": excluded_invalid_metrics,
            "metrics": metric_summary,
        },
        "characters": compared,
    }


def save_comparison_json(report: Dict, output_path: str) -> str:
    """Save a comparison report as JSON."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path


def save_comparison_csv(report: Dict, output_path: str) -> str:
    """Save paired character metrics in a flat, spreadsheet-friendly CSV."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["char", "before_image_count", "after_image_count"]
    for metric in METRIC_KEYS:
        fieldnames.extend((f"before_{metric}", f"after_{metric}", f"delta_{metric}"))

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for char, item in report["characters"].items():
            row = {
                "char": char,
                "before_image_count": item["before"]["image_count"],
                "after_image_count": item["after"]["image_count"],
            }
            for metric in METRIC_KEYS:
                row[f"before_{metric}"] = item["before"][metric]
                row[f"after_{metric}"] = item["after"][metric]
                row[f"delta_{metric}"] = item["delta"][metric]
            writer.writerow(row)
    return output_path


def print_comparison(report: Dict, detailed: bool = False) -> None:
    """Print a compact comparison summary."""
    info = report["comparison"]
    summary = report["summary"]
    print(f"\nPaired comparison: {info['before_dataset']} -> {info['after_dataset']}")
    print(f"Direction: {info['direction']}")
    print(
        f"Common characters: {summary['common_characters']} | "
        f"Compared: {summary['compared_characters']} | "
        f"Minimum samples per side: {info['min_samples']}"
    )
    print("\nMetric   Before    After     Mean delta  After higher")
    print("-" * 60)
    for metric in METRIC_KEYS:
        item = summary["metrics"][metric]
        if item["mean_delta"] is None:
            print(f"{metric:<8} No comparable values")
            continue
        print(
            f"{metric:<8} {item['before_mean']:<9.4f} {item['after_mean']:<9.4f} "
            f"{item['mean_delta']:<11.4f} {item['after_higher_rate']:<.1%}"
        )

    if detailed and report["characters"]:
        ranked = sorted(
            report["characters"].items(),
            key=lambda pair: pair[1]["delta"]["LQI"],
        )
        print("\nLargest LQI decreases:")
        for char, item in ranked[:10]:
            print(f"  {char}: {item['delta']['LQI']:+.4f}")
        print("Largest LQI increases:")
        for char, item in reversed(ranked[-10:]):
            print(f"  {char}: {item['delta']['LQI']:+.4f}")


def save_comparison_visualizations(report: Dict, output_dir: str) -> Iterable[str]:
    """Save mean-metric, LQI-ranking, and LQI-distribution charts."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if not report["characters"]:
        raise ValueError("No comparable characters available for visualization")

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in (
        "Arial Unicode MS",
        "Hiragino Sans GB",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ):
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    before_label = report["comparison"]["before_dataset"]
    after_label = report["comparison"]["after_dataset"]
    summary_metrics = report["summary"]["metrics"]
    output_paths = []

    x = np.arange(len(METRIC_KEYS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - width / 2,
        [summary_metrics[key]["before_mean"] for key in METRIC_KEYS],
        width,
        label=before_label,
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        [summary_metrics[key]["after_mean"] for key in METRIC_KEYS],
        width,
        label=after_label,
        color="#f58518",
    )
    ax.set_xticks(x, METRIC_KEYS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Paired mean score")
    ax.set_title("Paired metric comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_metric_means.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    ranked = sorted(
        report["characters"].items(),
        key=lambda pair: pair[1]["delta"]["LQI"],
    )
    selected = ranked[:10] + ranked[-10:]
    labels = [char for char, _ in selected]
    values = [item["delta"]["LQI"] for _, item in selected]
    colors = ["#d62728" if value < 0 else "#2ca02c" for value in values]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(selected)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(selected)), labels)
    ax.set_ylabel("Delta LQI (after - before)")
    ax.set_title("Largest paired LQI changes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "lqi_largest_changes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    lqi_deltas = [item["delta"]["LQI"] for item in report["characters"].values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lqi_deltas, bins=30, color="#72b7b2", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(fmean(lqi_deltas), color="#e45756", linestyle="--", label="Mean delta")
    ax.set_xlabel("Delta LQI (after - before)")
    ax.set_ylabel("Number of characters")
    ax.set_title("Distribution of paired LQI changes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "lqi_delta_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    return output_paths
