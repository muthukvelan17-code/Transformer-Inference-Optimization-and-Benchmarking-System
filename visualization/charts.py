"""
Matplotlib chart generation for benchmark analysis with modern AI infrastructure dark theme.
"""
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from config import DARK_THEME_COLORS


def apply_dark_theme(ax: plt.Axes, fig: plt.Figure):
    """Apply consistent sleek dark aesthetic to matplotlib axes and figure."""
    fig.patch.set_facecolor(DARK_THEME_COLORS["background"])
    ax.set_facecolor(DARK_THEME_COLORS["card_background"])

    # Spines
    for spine in ax.spines.values():
        spine.set_color(DARK_THEME_COLORS["grid"])
        spine.set_linewidth(1.0)

    # Ticks & Labels
    ax.tick_params(colors=DARK_THEME_COLORS["muted_text"], labelsize=10)
    ax.xaxis.label.set_color(DARK_THEME_COLORS["text"])
    ax.yaxis.label.set_color(DARK_THEME_COLORS["text"])
    ax.title.set_color(DARK_THEME_COLORS["text"])

    # Grid
    ax.grid(True, linestyle="--", alpha=0.3, color=DARK_THEME_COLORS["grid"])


def plot_latency_by_precision(runs: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """
    Bar chart of Mean & p95 latency (ms) across precision modes.
    """
    valid_runs = {k: v for k, v in runs.items() if v.get("status") == "SUCCESS"}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    apply_dark_theme(ax, fig)

    if not valid_runs:
        ax.text(
            0.5, 0.5, "No completed benchmark runs available",
            ha="center", va="center", color=DARK_THEME_COLORS["muted_text"], fontsize=12
        )
        ax.set_title("Inference Latency by Precision (ms)", fontsize=13, fontweight="bold", pad=12)
        return fig

    labels = list(valid_runs.keys())
    means = [valid_runs[k].get("mean_latency_ms", 0.0) for k in labels]
    p95s = [valid_runs[k].get("p95_latency_ms", 0.0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    bar1 = ax.bar(x - width/2, means, width, label="Mean Latency", color=DARK_THEME_COLORS["accent"], alpha=0.9, edgecolor="none", zorder=3)
    bar2 = ax.bar(x + width/2, p95s, width, label="p95 Latency", color=DARK_THEME_COLORS["purple"], alpha=0.9, edgecolor="none", zorder=3)

    # Value labels on top of bars
    for bar in bar1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=DARK_THEME_COLORS["text"], fontsize=9, fontweight="bold")
    for bar in bar2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=DARK_THEME_COLORS["text"], fontsize=9, fontweight="bold")

    ax.set_ylabel("Latency (milliseconds)", fontsize=11, fontweight="medium")
    ax.set_title("Inference Latency by Precision (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.legend(facecolor=DARK_THEME_COLORS["card_background"], edgecolor=DARK_THEME_COLORS["grid"],
              labelcolor=DARK_THEME_COLORS["text"], loc="upper right")
    fig.tight_layout()
    return fig


def plot_throughput_by_precision(runs: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """
    Bar chart of Throughput (samples/second) across precision modes.
    """
    valid_runs = {k: v for k, v in runs.items() if v.get("status") == "SUCCESS"}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    apply_dark_theme(ax, fig)

    if not valid_runs:
        ax.text(
            0.5, 0.5, "No completed benchmark runs available",
            ha="center", va="center", color=DARK_THEME_COLORS["muted_text"], fontsize=12
        )
        ax.set_title("Inference Throughput (samples/sec)", fontsize=13, fontweight="bold", pad=12)
        return fig

    labels = list(valid_runs.keys())
    throughputs = [valid_runs[k].get("throughput_samples_per_sec", 0.0) for k in labels]
    colors = [DARK_THEME_COLORS["primary"], DARK_THEME_COLORS["success"], DARK_THEME_COLORS["accent"]]

    bars = ax.bar(labels, throughputs, color=colors[:len(labels)], width=0.45, zorder=3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=DARK_THEME_COLORS["text"], fontsize=10, fontweight="bold")

    ax.set_ylabel("Throughput (samples / sec)", fontsize=11, fontweight="medium")
    ax.set_title("Inference Throughput (Higher is Better)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_memory_by_precision(runs: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """
    Bar chart of Memory Consumption (MB) across precision modes.
    """
    valid_runs = {k: v for k, v in runs.items() if v.get("status") == "SUCCESS"}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    apply_dark_theme(ax, fig)

    if not valid_runs:
        ax.text(
            0.5, 0.5, "No memory data available",
            ha="center", va="center", color=DARK_THEME_COLORS["muted_text"], fontsize=12
        )
        ax.set_title("Process Memory Consumption (MB)", fontsize=13, fontweight="bold", pad=12)
        return fig

    labels = list(valid_runs.keys())
    mems = [valid_runs[k].get("memory_rss_mb", 0.0) for k in labels]

    bars = ax.bar(labels, mems, color=DARK_THEME_COLORS["warning"], width=0.45, zorder=3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=DARK_THEME_COLORS["text"], fontsize=10, fontweight="bold")

    ax.set_ylabel("Process Memory RSS (MB)", fontsize=11, fontweight="medium")
    ax.set_title("Process Memory RSS by Precision Mode", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_batch_size_scaling(sweep_results: List[Dict[str, Any]]) -> plt.Figure:
    """
    Dual-axis line chart illustrating Latency and Throughput scaling across batch sizes.
    """
    valid = [r for r in sweep_results if r.get("status") == "SUCCESS"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    apply_dark_theme(ax1, fig)

    if not valid:
        ax1.text(
            0.5, 0.5, "No batch scaling data available",
            ha="center", va="center", color=DARK_THEME_COLORS["muted_text"], fontsize=12
        )
        ax1.set_title("Batch Size Scaling Analysis", fontsize=13, fontweight="bold", pad=12)
        return fig

    batch_sizes = [r.get("batch_size") for r in valid]
    latencies = [r.get("mean_latency_ms") for r in valid]
    throughputs = [r.get("throughput_samples_per_sec") for r in valid]

    # Latency line on ax1
    line1 = ax1.plot(batch_sizes, latencies, marker="o", linewidth=2.5,
                     color=DARK_THEME_COLORS["accent"], label="Latency (ms)", zorder=4)
    ax1.set_xlabel("Batch Size", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Mean Latency (ms)", color=DARK_THEME_COLORS["accent"], fontsize=11, fontweight="medium")
    ax1.tick_params(axis="y", labelcolor=DARK_THEME_COLORS["accent"])
    ax1.set_xticks(batch_sizes)

    # Throughput line on twin ax2
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    line2 = ax2.plot(batch_sizes, throughputs, marker="s", linewidth=2.5, linestyle="--",
                     color=DARK_THEME_COLORS["success"], label="Throughput (samples/s)", zorder=4)
    ax2.set_ylabel("Throughput (samples/s)", color=DARK_THEME_COLORS["success"], fontsize=11, fontweight="medium")
    ax2.tick_params(axis="y", labelcolor=DARK_THEME_COLORS["success"])
    for spine in ax2.spines.values():
        spine.set_color(DARK_THEME_COLORS["grid"])

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, facecolor=DARK_THEME_COLORS["card_background"],
               edgecolor=DARK_THEME_COLORS["grid"], labelcolor=DARK_THEME_COLORS["text"], loc="center left")

    ax1.set_title("Batch Size Scaling: Latency vs. Throughput", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_iteration_distribution(runs: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """
    Boxplot showing latency variance and outliers across benchmark iterations.
    """
    valid = {k: v for k, v in runs.items() if v.get("status") == "SUCCESS" and v.get("iteration_latencies_ms")}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    apply_dark_theme(ax, fig)

    if not valid:
        ax.text(
            0.5, 0.5, "No iteration data available",
            ha="center", va="center", color=DARK_THEME_COLORS["muted_text"], fontsize=12
        )
        ax.set_title("Iteration Latency Variance", fontsize=13, fontweight="bold", pad=12)
        return fig

    labels = list(valid.keys())
    data = [valid[k]["iteration_latencies_ms"] for k in labels]

    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor=DARK_THEME_COLORS["card_background"], color=DARK_THEME_COLORS["accent"], linewidth=1.5),
        capprops=dict(color=DARK_THEME_COLORS["muted_text"]),
        whiskerprops=dict(color=DARK_THEME_COLORS["muted_text"]),
        flierprops=dict(marker="o", color=DARK_THEME_COLORS["warning"], alpha=0.6),
        medianprops=dict(color=DARK_THEME_COLORS["success"], linewidth=2.0),
    )

    ax.set_ylabel("Latency (ms)", fontsize=11, fontweight="medium")
    ax.set_title("Iteration Latency Distribution (Stability)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
