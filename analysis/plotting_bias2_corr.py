
"""Correlation analysis between stain-level and class-level metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
CLASS_PREFIX = "class_"


@dataclass
class MetricPair:
    """Representation of a stain/class metric pair that share the same suffix."""

    base_metric: str
    stain_column: str
    class_column: str


@dataclass
class MetricCorrelation:
    """Computed correlation statistics for a single metric pair."""

    base_metric: str
    stain_column: str
    class_column: str
    correlation: float
    sample_size: int


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

def discover_metric_pairs(df: pd.DataFrame) -> list[MetricPair]:
    """Find matching stain/class metric pairs based on shared suffix names."""

    stain_columns = [column for column in df.columns if column.startswith(STAIN_PREFIX)]
    class_columns = [column for column in df.columns if column.startswith(CLASS_PREFIX)]

    class_lookup = {column[len(CLASS_PREFIX) :]: column for column in class_columns}

    pairs: list[MetricPair] = []
    for stain_column in stain_columns:
        base_metric = stain_column[len(STAIN_PREFIX) :]
        class_column = class_lookup.get(base_metric)
        if class_column is not None:
            pairs.append(MetricPair(base_metric, stain_column, class_column))

    return pairs


def find_metric_pair(pairs: Iterable[MetricPair], base_metric: str) -> MetricPair | None:
    """Locate the metric pair whose suffix matches ``base_metric``."""

    target = base_metric.lower()
    for pair in pairs:
        if pair.base_metric.lower() == target:
            return pair
    return None


def find_metric_correlation(
    correlations: Iterable[MetricCorrelation], base_metric: str
) -> MetricCorrelation | None:
    """Return the correlation entry that matches ``base_metric``."""

    target = base_metric.lower()
    for entry in correlations:
        if entry.base_metric.lower() == target:
            return entry
    return None


def compute_metric_correlations(
    df: pd.DataFrame, pairs: Iterable[MetricPair]
) -> list[MetricCorrelation]:
    """Compute correlations for each stain/class metric pair."""

    results: list[MetricCorrelation] = []

    for pair in pairs:
        subset = df[[pair.stain_column, pair.class_column]].dropna()
        sample_size = len(subset)

        if sample_size < 2:
            correlation = np.nan
        else:
            correlation = subset[pair.stain_column].corr(subset[pair.class_column])

        results.append(
            MetricCorrelation(
                base_metric=pair.base_metric,
                stain_column=pair.stain_column,
                class_column=pair.class_column,
                correlation=correlation,
                sample_size=sample_size,
            )
        )

    def sort_key(item: MetricCorrelation) -> tuple[int, float]:
        if np.isnan(item.correlation):
            return (0, 0.0)
        return (1, abs(item.correlation))

    results.sort(key=sort_key, reverse=True)
    return results


def _format_metric_label(column_name: str) -> str:
    """Human readable label for a metric column."""

    if column_name.startswith(STAIN_PREFIX):
        prefix = "Stain"
        base = column_name[len(STAIN_PREFIX) :]
    elif column_name.startswith(CLASS_PREFIX):
        prefix = "Class"
        base = column_name[len(CLASS_PREFIX) :]
    else:
        prefix = ""
        base = column_name

    base = base.replace("_", " ").strip()
    if base:
        base = base.title()

    return f"{prefix} {base}".strip()


def _format_base_metric_name(base_metric: str) -> str:
    formatted = base_metric.replace("_", " ").strip()
    return formatted.title() if formatted else base_metric


def build_correlation_table(correlations: Iterable[MetricCorrelation]) -> pd.DataFrame:
    """Create a tabular view of correlation statistics."""

    records = []
    for entry in correlations:
        records.append(
            {
                "Metric": _format_base_metric_name(entry.base_metric),
                "Correlation": entry.correlation,
                "Samples": entry.sample_size,
                "Stain Metric": entry.stain_column,
                "Class Metric": entry.class_column,
            }
        )

    return pd.DataFrame(records)


def plot_metric_pair(
    pdf: PdfPages,
    df: pd.DataFrame,
    pair: MetricPair | None,
    *,
    correlation: MetricCorrelation | None = None,
    title: str | None = None,
) -> None:
    """Plot a scatter chart for a specific stain/class metric pair."""

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    if pair is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Requested stain/class metric pair was not found in the dataset.",
            ha="center",
            va="center",
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)
        return

    required_columns = [pair.stain_column, pair.class_column]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "The dataset does not contain all columns required for this plot.",
            ha="center",
            va="center",
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)
        return

    subset = df[required_columns].dropna()

    if subset.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Insufficient data to render scatter plot for this metric pair.",
            ha="center",
            va="center",
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)
        return

    x = subset[pair.stain_column]
    y = subset[pair.class_column]

    ax.scatter(x, y, alpha=0.7, edgecolor="black", linewidth=0.5)

    if x.nunique() > 1 and y.nunique() > 1:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=1.5)

    if correlation is not None and not np.isnan(correlation.correlation):
        corr_value = correlation.correlation
        sample_size = correlation.sample_size
    else:
        sample_size = len(subset)
        if sample_size >= 2:
            corr_value = subset[pair.stain_column].corr(subset[pair.class_column])
        else:
            corr_value = np.nan

    metric_title = title or f"Stain vs Class: {_format_base_metric_name(pair.base_metric)}"
    if np.isnan(corr_value):
        stats_text = f"Pearson r unavailable (n={sample_size})"
    else:
        stats_text = f"Pearson r = {corr_value:.3f} (n={sample_size})"

    ax.set_xlabel(_format_metric_label(pair.stain_column))
    ax.set_ylabel(_format_metric_label(pair.class_column))
    ax.set_title(f"{metric_title}\n{stats_text}", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

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


def render_correlation_table(pdf: PdfPages, table_df: pd.DataFrame, max_rows: int = 20) -> None:
    """Add a table of correlations to the PDF report."""

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.set_title(
        "Correlation Between Matching Stain/Class Metrics",
        fontsize=16,
        pad=20,
    )

    if table_df.empty:
        ax.text(
            0.5,
            0.5,
            "No overlapping stain/class metrics found in the dataset.",
            ha="center",
            va="center",
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)
        return

    display_df = table_df.copy()
    display_df["Correlation"] = display_df["Correlation"].apply(
        lambda value: "–" if pd.isna(value) else f"{value:.3f}"
    )
    display_df["Samples"] = display_df["Samples"].astype(int)

    if len(display_df) > max_rows:
        note = f"Showing top {max_rows} of {len(display_df)} metric pairs sorted by |correlation|."
        display_df = display_df.head(max_rows)
    else:
        note = None

    cell_text = display_df.values.tolist()
    column_labels = list(display_df.columns)

    table = ax.table(cellText=cell_text, colLabels=column_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)

    if note:
        ax.text(0.5, 0.05, note, ha="center", va="center", fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def plot_best_correlation(
    pdf: PdfPages,
    df: pd.DataFrame,
    correlation: MetricCorrelation | None,
) -> None:
    """Plot a scatter chart for the strongest correlation."""

    if correlation is None or np.isnan(correlation.correlation):
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No valid correlations available for plotting.",
            ha="center",
            va="center",
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)
        return

    pair = MetricPair(
        base_metric=correlation.base_metric,
        stain_column=correlation.stain_column,
        class_column=correlation.class_column,
    )

    plot_metric_pair(
        pdf,
        df,
        pair,
        correlation=correlation,
        title=f"Best Matching Metric Pair: {_format_base_metric_name(pair.base_metric)}",
    )

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

    correlations: list[MetricCorrelation],
    pairs: Iterable[MetricPair],
    output_path: Path,
) -> None:
    """Generate the PDF report with the correlation table and scatter plot."""

    ensure_output_directory(output_path)

    table_df = build_correlation_table(correlations)
    strongest = correlations[0] if correlations else None

    with PdfPages(output_path) as pdf:
        render_correlation_table(pdf, table_df)
        plot_best_correlation(pdf, df, strongest)

        for base_metric in ("f1", "auc"):
            pair = find_metric_pair(pairs, base_metric)
            correlation = find_metric_correlation(correlations, base_metric)
            plot_metric_pair(
                pdf,
                df,
                pair,
                correlation=correlation,
                title=f"Stain vs Class Comparison: {_format_base_metric_name(base_metric)}",
            )

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

    metric_pairs = discover_metric_pairs(df)
    correlations = compute_metric_correlations(df, metric_pairs)
    create_bias2_corr_report(df, correlations, metric_pairs, OUTPUT_PATH)

    table_df = build_correlation_table(correlations)
    if table_df.empty:
        print("No overlapping stain/class metrics were found in the dataset.")
    else:
        print("Correlation between matching stain/class metrics:")
        print(table_df.to_string(index=False, na_rep="NaN", float_format=lambda v: f"{v:.3f}"))

    stain_metrics, class_metrics = identify_metric_columns(df)
    pairwise_stats = collect_pairwise_correlations(df, stain_metrics, class_metrics)
    create_bias2_corr_report(df, stain_metrics, class_metrics, pairwise_stats, OUTPUT_PATH)

    print(f"Saved correlation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
