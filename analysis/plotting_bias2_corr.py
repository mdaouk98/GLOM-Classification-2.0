"""Generate correlation plots between stain-level and class-level metrics.

This module reads the ``bias2_corr`` detailed evaluation CSV, computes
correlation statistics between overall stain metrics and per-class metrics, and
exports several visualizations into a single multi-page PDF report.

Usage
-----
Run the script directly:

```
python -m analysis.plotting_bias2_corr
```

The resulting PDF is saved to ``analysis/plots/bias2_corr_plots.pdf``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DATA_PATH = Path("analysis/tables/bias2_corr_model_evaluation_detailed.csv")
OUTPUT_PATH = Path("analysis/plots/bias2_corr_plots.pdf")
STAIN_PREFIX = "stain_"
CLASS_TOKEN = "_class_"


def load_bias2_corr_results(csv_path: Path) -> pd.DataFrame:
    """Load the Bias2 correlation CSV file and coerce numeric columns."""

    if not csv_path.exists():  # pragma: no cover - runtime validation
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in df.columns:
        if column == "Model":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Fold" in df.columns:
        df["Fold"] = df["Fold"].astype(pd.Int64Dtype())

    return df


def ensure_output_directory(output_path: Path) -> None:
    """Create the parent directory of ``output_path`` if required."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def identify_metric_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return lists of stain-level metrics and class-level metrics."""

    stain_metrics: list[str] = []
    class_metrics: list[str] = []

    for column in df.columns:
        if column in {"Model", "Fold"}:
            continue
        if column.startswith(STAIN_PREFIX) and CLASS_TOKEN not in column:
            stain_metrics.append(column)
        elif column.startswith("class_") or CLASS_TOKEN in column:
            class_metrics.append(column)

    return stain_metrics, class_metrics


def _format_metric_name(name: str) -> str:
    return name.replace("stain_", "").replace("class_", "Class ").replace("_", " ").title()


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render a heatmap with annotated values."""

    if data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    im = ax.imshow(data.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([_format_metric_name(name) for name in data.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([_format_metric_name(name) for name in data.index])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if pd.isna(value):
                label = "N/A"
            else:
                label = f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def compute_correlation_matrix(
    df: pd.DataFrame,
    row_metrics: Sequence[str],
    column_metrics: Sequence[str],
) -> pd.DataFrame:
    """Compute the Pearson correlation matrix between metric groups."""

    available_rows = [metric for metric in row_metrics if metric in df]
    available_cols = [metric for metric in column_metrics if metric in df]

    if not available_rows or not available_cols:
        return pd.DataFrame(index=available_rows, columns=available_cols, dtype=float)

    relevant = df[available_rows + available_cols].replace([np.inf, -np.inf], np.nan)
    corr = relevant.corr(method="pearson")
    return corr.loc[available_rows, available_cols]


def collect_pairwise_correlations(
    df: pd.DataFrame,
    stain_metrics: Sequence[str],
    class_metrics: Sequence[str],
) -> list[tuple[float, float, str, str, int]]:
    """Return correlation statistics for all metric pairs."""

    pairs: list[tuple[float, float, str, str, int]] = []

    for stain_metric in stain_metrics:
        if stain_metric not in df:
            continue
        for class_metric in class_metrics:
            if class_metric not in df:
                continue
            subset = df[[stain_metric, class_metric]].dropna()
            sample_size = len(subset)
            if sample_size < 2:
                continue
            corr_value = subset[stain_metric].corr(subset[class_metric])
            pairs.append((abs(corr_value), corr_value, stain_metric, class_metric, sample_size))

    pairs.sort(key=lambda entry: entry[0], reverse=True)
    return pairs


def _format_correlation_line(
    corr_value: float,
    stain_metric: str,
    class_metric: str,
    sample_size: int,
) -> str:
    arrow = "↑" if corr_value >= 0 else "↓"
    return (
        f"{_format_metric_name(stain_metric)} vs. {_format_metric_name(class_metric)}: "
        f"{corr_value:.3f} ({arrow}, n={sample_size})"
    )


def plot_summary_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    stain_metrics: Sequence[str],
    class_metrics: Sequence[str],
    pairwise_stats: Sequence[tuple[float, float, str, str, int]],
) -> None:
    """Render an introductory page summarizing the dataset."""

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        "Bias2 Correlation Analysis",
        "",
        f"Total records: {len(df)}",
        f"Models evaluated: {df['Model'].nunique() if 'Model' in df else 'N/A'}",
        f"Available stain metrics: {len(stain_metrics)}",
        f"Available class metrics: {len(class_metrics)}",
        "",
        "Top correlated metric pairs:",
    ]

    top_pairs = list(pairwise_stats[:5])
    if not top_pairs:
        lines.append("  (No correlations could be computed.)")
    else:
        for _, corr_value, stain_metric, class_metric, sample_size in top_pairs:
            lines.append(
                "  "
                + _format_correlation_line(corr_value, stain_metric, class_metric, sample_size)
            )

    negative_pairs = [entry for entry in pairwise_stats if entry[1] < 0]
    lines.append("")
    lines.append("Strongest negative correlations:")
    if not negative_pairs:
        lines.append("  (No negative correlations identified.)")
    else:
        for _, corr_value, stain_metric, class_metric, sample_size in negative_pairs[:5]:
            lines.append(
                "  "
                + _format_correlation_line(corr_value, stain_metric, class_metric, sample_size)
            )

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=12)

    pdf.savefig(fig)
    plt.close(fig)


def plot_overall_correlation_heatmap(
    pdf: PdfPages,
    df: pd.DataFrame,
    stain_metrics: Sequence[str],
    class_metrics: Sequence[str],
) -> None:
    """Plot the correlation heatmap aggregated across all models."""

    if not stain_metrics or not class_metrics:
        return

    heatmap_data = compute_correlation_matrix(df, stain_metrics, class_metrics)

    fig_height = max(4.0, 0.5 * len(stain_metrics) + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    _draw_heatmap(ax, heatmap_data, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title("Overall Correlation Between Stain Metrics and Class Metrics")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_model_specific_heatmaps(
    pdf: PdfPages,
    df: pd.DataFrame,
    stain_metrics: Sequence[str],
    class_metrics: Sequence[str],
) -> None:
    """Create per-model correlation heatmaps."""

    if "Model" not in df.columns or not stain_metrics or not class_metrics:
        return

    models = sorted(df["Model"].dropna().unique())
    if not models:
        return

    ncols = 2
    nrows = 2
    per_page = ncols * nrows

    for start in range(0, len(models), per_page):
        subset_models = models[start : start + per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)
        for ax in axes.flatten():
            ax.set_visible(False)

        for ax, model in zip(axes.flatten(), subset_models):
            ax.set_visible(True)
            model_df = df[df["Model"] == model]
            heatmap_data = compute_correlation_matrix(model_df, stain_metrics, class_metrics)
            _draw_heatmap(ax, heatmap_data, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            ax.set_title(str(model))

        fig.suptitle("Correlation by Model")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)


def plot_top_correlation_scatters(
    pdf: PdfPages,
    df: pd.DataFrame,
    pairwise_stats: Sequence[tuple[float, float, str, str, int]],
    max_plots: int = 6,
) -> None:
    """Scatter plots for the strongest correlations."""

    top_pairs = list(pairwise_stats[:max_plots])
    if not top_pairs:
        return

    ncols = 2
    nrows = math.ceil(len(top_pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)

    for ax in axes.flatten()[len(top_pairs) :]:
        ax.set_visible(False)

    for ax, (_, corr_value, stain_metric, class_metric, sample_size) in zip(axes.flatten(), top_pairs):
        subset = df[[stain_metric, class_metric]].dropna()
        ax.scatter(subset[stain_metric], subset[class_metric], alpha=0.7, edgecolor="black", linewidth=0.5)

        if len(subset) >= 2:
            x = subset[stain_metric].to_numpy()
            y = subset[class_metric].to_numpy()
            try:
                slope, intercept = np.polyfit(x, y, deg=1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1.5)
            except np.linalg.LinAlgError:
                pass

        ax.set_xlabel(_format_metric_name(stain_metric))
        ax.set_ylabel(_format_metric_name(class_metric))
        ax.set_title(
            f"r = {corr_value:.3f} (n={sample_size})",
            fontsize=10,
        )
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Top Correlated Metric Pairs")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def create_bias2_corr_report(
    df: pd.DataFrame,
    stain_metrics: Sequence[str],
    class_metrics: Sequence[str],
    pairwise_stats: Sequence[tuple[float, float, str, str, int]],
    output_path: Path,
) -> None:
    """Generate the multi-page PDF correlation report."""

    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        plot_summary_page(pdf, df, stain_metrics, class_metrics, pairwise_stats)
        plot_overall_correlation_heatmap(pdf, df, stain_metrics, class_metrics)
        plot_model_specific_heatmaps(pdf, df, stain_metrics, class_metrics)
        plot_top_correlation_scatters(pdf, df, pairwise_stats)


def main() -> None:
    df = load_bias2_corr_results(DATA_PATH)
    stain_metrics, class_metrics = identify_metric_columns(df)
    pairwise_stats = collect_pairwise_correlations(df, stain_metrics, class_metrics)
    create_bias2_corr_report(df, stain_metrics, class_metrics, pairwise_stats, OUTPUT_PATH)
    print(f"Saved correlation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
