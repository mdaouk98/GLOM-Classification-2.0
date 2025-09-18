"""Generate plots summarizing bias evaluation metrics.

This module reads the detailed bias evaluation CSV file produced by
``analysis/main_table_bias2.py`` and generates a multi-page PDF report with
common visualizations.  The generated PDF is stored under
``analysis/plots/bias2_plots.pdf``.

Usage
-----
Run the script directly:

```
python -m analysis.plotting_bias2
```

The script will read the CSV file, build several summary figures and export
all of them into a single PDF file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DATA_PATH = Path("analysis/tables/bias2_model_evaluation_detailed.csv")
OUTPUT_PATH = Path("analysis/plots/bias2_plots.pdf")
DEFAULT_METRICS: Sequence[str] = (
    "stain_Accuracy",
    "stain_Precision",
    "stain_Recall",
    "stain_F1",
    "stain_AUC",
)
CLASS_ACCURACY_PREFIX = "stain_Acc_class_"


def load_bias2_results(csv_path: Path) -> pd.DataFrame:
    """Load the Bias2 evaluation CSV file and ensure numeric dtypes."""

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
    """Create the parent directory of ``output_path`` if necessary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def determine_model_order(df: pd.DataFrame, metric: str) -> list[str]:
    """Return model names sorted by descending average ``metric``."""

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not present in the dataframe")

    averages = (
        df.groupby("Model")[metric]
        .mean()
        .sort_values(ascending=False)
    )
    return averages.index.tolist()


def _format_metric_name(metric: str) -> str:
    return metric.replace("stain_", "").replace("_", " ").title()


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render a heatmap with value annotations using matplotlib only."""

    if data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    matrix = data.to_numpy()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([_format_metric_name(name) for name in data.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if pd.isna(value):
                text = "N/A"
                color = "black"
            else:
                text = f"{value:.3f}"
                norm_value = im.norm(value)
                color = "white" if norm_value > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_summary_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
) -> None:
    """Add a summary text page with key statistics."""

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    num_models = len(model_order)
    num_records = len(df)

    lines = [
        "Bias2 Evaluation Overview",
        "",
        f"Total records: {num_records}",
        f"Models evaluated: {num_models}",
        "",
    ]

    if metrics:
        averages = df.groupby("Model")[list(metrics)].mean()
        for metric in metrics:
            if metric not in averages:
                continue
            best_model = averages[metric].idxmax()
            best_value = averages.loc[best_model, metric]
            worst_model = averages[metric].idxmin()
            worst_value = averages.loc[worst_model, metric]
            lines.append(
                f"Best {_format_metric_name(metric)}: {best_model} ({best_value:.3f})"
            )
            lines.append(
                f"Lowest {_format_metric_name(metric)}: {worst_model} ({worst_value:.3f})"
            )
            lines.append("")

    text = "\n".join(lines).strip()
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=12)

    pdf.savefig(fig)
    plt.close(fig)


def plot_average_metric_heatmap(
    pdf: PdfPages,
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
) -> None:
    """Plot the average metrics per model as a heatmap."""

    if not metrics:
        return

    summary = df.groupby("Model")[list(metrics)].mean()
    summary = summary.reindex(model_order)

    fig_height = max(4.0, 0.6 * len(summary.index))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    _draw_heatmap(ax, summary, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Average Performance Metrics Per Model")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_class_accuracy_heatmap(
    pdf: PdfPages,
    df: pd.DataFrame,
    class_columns: Sequence[str],
    model_order: Sequence[str],
) -> None:
    """Plot average per-class accuracy as a heatmap."""

    if not class_columns:
        return

    summary = df.groupby("Model")[list(class_columns)].mean()
    summary = summary.reindex(model_order)

    fig_height = max(4.0, 0.6 * len(summary.index))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    _draw_heatmap(ax, summary, cmap="PuBuGn", vmin=0.0, vmax=1.0)
    ax.set_title("Average Per-Class Accuracy Per Model")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_metric_boxplots(
    pdf: PdfPages,
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
) -> None:
    """Plot per-fold distributions for each metric as boxplots."""

    if not metrics:
        return

    n_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=False)
    axes = axes.flatten()
    palette = plt.get_cmap("tab20c")(np.linspace(0.05, 0.95, len(model_order)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        series_per_model = [
            df.loc[df["Model"] == model, metric].dropna().to_numpy()
            for model in model_order
        ]

        if all(len(values) == 0 for values in series_per_model):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        boxplot = ax.boxplot(
            series_per_model,
            labels=model_order,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        for patch, color in zip(boxplot["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in boxplot["medians"]:
            median.set_color("black")
        for mean in boxplot["means"]:
            mean.set_color("red")

        ax.set_title(f"Distribution of {_format_metric_name(metric)}")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[n_metrics:]:
        ax.remove()

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_accuracy_trend(
    pdf: PdfPages,
    df: pd.DataFrame,
    model_order: Sequence[str],
) -> None:
    """Plot accuracy across folds for each model."""

    if "Fold" not in df.columns or "stain_Accuracy" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, model in enumerate(model_order):
        subset = df[df["Model"] == model].sort_values("Fold")
        if subset.empty:
            continue
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.plot(
            subset["Fold"],
            subset["stain_Accuracy"],
            marker="o",
            label=model,
            color=color,
        )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Stain Accuracy")
    ax.set_title("Fold-wise Accuracy Per Model")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_uncertainty_summary(
    pdf: PdfPages,
    df: pd.DataFrame,
    model_order: Sequence[str],
) -> None:
    """Plot the average predictive uncertainty per model as a bar chart."""

    if "stain_Uncertainty" not in df.columns:
        return

    summary = df.groupby("Model")["stain_Uncertainty"].agg(["mean", "std"])
    summary = summary.reindex(model_order)

    fig_height = max(4.0, 0.5 * len(summary.index) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#5ab4ac", alpha=0.8)
    ax.set_xlabel("Average Predictive Entropy")
    ax.set_title("Uncertainty Summary Per Model")
    ax.invert_yaxis()

    for y_pos, (mean_val, std_val) in enumerate(zip(summary["mean"], summary["std"])):
        if pd.isna(mean_val):
            label = "N/A"
            mean_for_text = 0.0
        else:
            mean_for_text = float(mean_val)
            if pd.isna(std_val) or std_val == 0:
                label = f"{mean_val:.4f}"
            else:
                label = f"{mean_val:.4f} ± {std_val:.4f}"
        offset = 0.0 if pd.isna(std_val) else float(std_val)
        ax.text(
            mean_for_text + offset,
            y_pos,
            label,
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_bias2_report(
    df: pd.DataFrame,
    output_path: Path,
    metrics: Sequence[str],
    class_columns: Sequence[str],
    model_order: Sequence[str],
) -> None:
    """Generate the multi-page PDF report."""

    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        plot_summary_page(pdf, df, metrics, model_order)
        plot_average_metric_heatmap(pdf, df, metrics, model_order)
        plot_class_accuracy_heatmap(pdf, df, class_columns, model_order)
        plot_metric_boxplots(pdf, df, metrics, model_order)
        plot_accuracy_trend(pdf, df, model_order)
        plot_uncertainty_summary(pdf, df, model_order)


def main() -> None:
    df = load_bias2_results(DATA_PATH)

    metrics = [metric for metric in DEFAULT_METRICS if metric in df.columns]
    class_columns = sorted(
        (column for column in df.columns if column.startswith(CLASS_ACCURACY_PREFIX)),
        key=lambda name: int(name.rsplit("_", maxsplit=1)[-1]),
    )

    if metrics:
        metric_for_order = metrics[0]
    else:
        candidate_columns = [
            column
            for column in df.columns
            if column not in {"Model", "Fold"}
        ]
        if not candidate_columns:
            raise ValueError("No numeric metrics available to sort models.")
        metric_for_order = candidate_columns[0]
    model_order = determine_model_order(df, metric_for_order)

    create_bias2_report(df, OUTPUT_PATH, metrics, class_columns, model_order)
    print(f"Saved report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
