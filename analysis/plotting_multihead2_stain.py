"""Generate plots for the multi-head stain evaluation results.

This module focuses on three groups of metrics extracted from
``analysis/tables/multihead3_stain_model_evaluation_detailed.csv``:

* **class metrics** – overall model performance per classification head
* **stain metrics** – metrics computed for the stain prediction head
* **per-stain metrics** – metrics computed for each individual stain class

The script builds a collection of figures that highlight the behaviour of
these metric groups without mixing them in the same visualization.  The
figures are exported to ``analysis/plots/bias2_plots.pdf`` as a multi-page
report.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DATA_PATH = Path("analysis/tables/multihead3_stain_model_evaluation_detailed.csv")
OUTPUT_PATH = Path("analysis/plots/bias2_plots.pdf")
CLASS_SUMMARY_METRICS: Sequence[str] = (
    "class_Accuracy",
    "class_Precision",
    "class_Recall",
    "class_F1",
    "class_AUC",
    "class_Uncertainty",
)
STAIN_SUMMARY_METRICS: Sequence[str] = (
    "stain_Accuracy",
    "stain_Precision",
    "stain_Recall",
    "stain_F1",
    "stain_AUC",
    "stain_Uncertainty",
)
CLASS_PER_CLASS_PREFIX = "class_Acc_class_"
STAIN_PER_CLASS_PREFIX = "stain_Acc_class_"
PER_STAIN_PREFIX = "class_PerStain_"


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load the evaluation CSV and coerce numeric columns."""

    if not csv_path.exists():  # pragma: no cover - runtime validation
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in df.columns:
        if column in {"Model"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Fold" in df.columns:
        df["Fold"] = df["Fold"].astype(pd.Int64Dtype())

    return df


def ensure_output_directory(output_path: Path) -> None:
    """Create parent directories for the output if they do not exist."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def determine_model_order(df: pd.DataFrame, metric: str) -> List[str]:
    """Return model names sorted by descending average ``metric``."""

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not present in dataframe")

    averages = df.groupby("Model")[metric].mean().sort_values(ascending=False)
    return averages.index.tolist()


def _format_metric_name(metric: str) -> str:
    metric = metric.replace("class_", "").replace("stain_", "")
    metric = metric.replace("PerStain_", "per stain ")
    metric = metric.replace("Acc_class_", "Accuracy class ")
    metric = metric.replace("_", " ")
    return metric.title()


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    value_format: str = "{value:.3f}",
) -> None:
    """Render a simple heatmap with value annotations."""

    if data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    matrix = data.to_numpy()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.iloc[i, j]
            if pd.isna(value):
                text = "N/A"
                color = "black"
            else:
                text = value_format.format(value=float(value))
                norm_value = im.norm(value)
                color = "white" if norm_value > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_summary_page(pdf: PdfPages, df: pd.DataFrame, model_order: Sequence[str]) -> None:
    """Add a text-only summary page."""

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    lines = [
        "Multi-head Stain Evaluation Overview",
        "",
        f"Total records: {len(df)}",
        f"Models evaluated: {len(model_order)}",
        "",
    ]

    def _append_metric_summary(metrics: Iterable[str], title: str) -> None:
        metrics = [metric for metric in metrics if metric in df.columns]
        if not metrics:
            return
        lines.append(title)
        grouped = df.groupby("Model")[metrics].mean()
        for metric in metrics:
            best_model = grouped[metric].idxmax()
            best_value = grouped.loc[best_model, metric]
            lines.append(
                f"  Best {_format_metric_name(metric)}: {best_model} ({best_value:.3f})"
            )
        lines.append("")

    _append_metric_summary(CLASS_SUMMARY_METRICS, "Class metrics")
    _append_metric_summary(STAIN_SUMMARY_METRICS, "Stain metrics")

    text = "\n".join(lines).strip()
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=12)

    pdf.savefig(fig)
    plt.close(fig)


def plot_metric_heatmap(
    pdf: PdfPages,
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
    title: str,
) -> None:
    """Plot average metrics per model for the provided ``metrics`` list."""

    metrics = [metric for metric in metrics if metric in df.columns]
    if not metrics:
        return

    summary = df.groupby("Model")[list(metrics)].mean().reindex(model_order)
    display = summary.rename(columns=lambda name: _format_metric_name(name))

    data = summary.to_numpy(dtype=float)
    if np.isnan(data).all():
        vmax = 1.0
    else:
        vmax = float(np.nanmax(data))
        vmax = max(1.0, vmax)

    fig_height = max(4.0, 0.6 * len(summary.index))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    _draw_heatmap(ax, display, cmap="YlGnBu", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_class_per_class_heatmap(
    pdf: PdfPages,
    df: pd.DataFrame,
    prefix: str,
    model_order: Sequence[str],
    title: str,
) -> None:
    """Plot per-class accuracy metrics for a given prefix."""

    columns = sorted(
        (column for column in df.columns if column.startswith(prefix)),
        key=lambda name: int(name.rsplit("_", maxsplit=1)[-1]),
    )
    if not columns:
        return

    summary = df.groupby("Model")[columns].mean().reindex(model_order)
    renamed = summary.rename(columns=lambda name: name.rsplit("_", maxsplit=1)[-1])

    data = summary.to_numpy(dtype=float)
    if np.isnan(data).all():
        vmax = 1.0
    else:
        vmax = float(np.nanmax(data))
        vmax = max(1.0, vmax)

    fig_height = max(4.0, 0.6 * len(summary.index))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    _draw_heatmap(ax, renamed, cmap="PuBu", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_metric_boxplots(
    pdf: PdfPages,
    df: pd.DataFrame,
    metrics: Sequence[str],
    model_order: Sequence[str],
    title: str,
) -> None:
    """Plot boxplots for the selected metrics."""

    metrics = [metric for metric in metrics if metric in df.columns]
    if not metrics:
        return

    n_metrics = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=False)
    axes = np.atleast_1d(axes).flatten()
    palette = plt.get_cmap("tab20c")(np.linspace(0.1, 0.9, len(model_order)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [
            df.loc[df["Model"] == model, metric].dropna().to_numpy()
            for model in model_order
        ]
        if all(len(series) == 0 for series in values):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        boxplot = ax.boxplot(
            values,
            labels=model_order,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        for patch, color in zip(boxplot["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in boxplot["medians"]:
            median.set_color("black")
        for mean in boxplot["means"]:
            mean.set_color("red")

        ax.set_title(_format_metric_name(metric))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[n_metrics:]:
        ax.remove()

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def extract_per_stain_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Return mapping metric -> mapping[stain] = column name."""

    per_stain_columns: Dict[str, Dict[str, str]] = defaultdict(dict)
    for column in df.columns:
        if not column.startswith(PER_STAIN_PREFIX):
            continue
        rest = column[len(PER_STAIN_PREFIX) :]
        try:
            stain_id, metric = rest.split("_", 1)
        except ValueError:
            continue
        per_stain_columns[metric][stain_id] = column
    return per_stain_columns


def plot_per_stain_heatmaps(
    pdf: PdfPages,
    df: pd.DataFrame,
    per_stain_columns: Dict[str, Dict[str, str]],
    model_order: Sequence[str],
) -> None:
    """Plot a heatmap for each per-stain metric type."""

    for metric, stain_columns in sorted(per_stain_columns.items()):
        ordered_items = sorted(stain_columns.items(), key=lambda item: int(item[0]))
        columns = [column for _, column in ordered_items]
        summary = df.groupby("Model")[columns].mean().reindex(model_order)
        renamed = summary.rename(
            columns={column: f"Stain {stain_id}" for stain_id, column in ordered_items}
        )

        data = summary.to_numpy(dtype=float)
        if np.isnan(data).all():
            vmax = 1.0
        else:
            vmax = float(np.nanmax(data))
            vmax = max(1.0, vmax)

        fig_height = max(4.0, 0.6 * len(summary.index))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        _draw_heatmap(ax, renamed, cmap="OrRd", vmin=0.0, vmax=vmax)
        formatted_title = _format_metric_name(metric)
        ax.set_title(f"Per-Stain {formatted_title}")
        ax.set_xlabel("Stain")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def create_report(df: pd.DataFrame, output_path: Path) -> None:
    """Build the multi-page PDF report."""

    ensure_output_directory(output_path)

    metric_for_order = "class_Accuracy" if "class_Accuracy" in df.columns else None
    if metric_for_order is None:
        candidates = [
            column
            for column in df.columns
            if column not in {"Model", "Scale", "Cluster", "Fold"}
        ]
        if not candidates:
            raise ValueError("No numeric metrics available to order models.")
        metric_for_order = candidates[0]
    model_order = determine_model_order(df, metric_for_order)

    per_stain_columns = extract_per_stain_metrics(df)

    with PdfPages(output_path) as pdf:
        plot_summary_page(pdf, df, model_order)
        plot_metric_heatmap(
            pdf,
            df,
            CLASS_SUMMARY_METRICS,
            model_order,
            "Average Class Metrics Per Model",
        )
        plot_class_per_class_heatmap(
            pdf,
            df,
            CLASS_PER_CLASS_PREFIX,
            model_order,
            "Per-Class Accuracy For Class Head",
        )
        plot_metric_boxplots(
            pdf,
            df,
            CLASS_SUMMARY_METRICS,
            model_order,
            "Class Metric Distributions",
        )
        plot_metric_heatmap(
            pdf,
            df,
            STAIN_SUMMARY_METRICS,
            model_order,
            "Average Stain Metrics Per Model",
        )
        plot_class_per_class_heatmap(
            pdf,
            df,
            STAIN_PER_CLASS_PREFIX,
            model_order,
            "Per-Class Accuracy For Stain Head",
        )
        plot_metric_boxplots(
            pdf,
            df,
            STAIN_SUMMARY_METRICS,
            model_order,
            "Stain Metric Distributions",
        )
        plot_per_stain_heatmaps(pdf, df, per_stain_columns, model_order)

    print(f"Saved report to {output_path}")


def main() -> None:
    df = load_results(DATA_PATH)
    create_report(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
