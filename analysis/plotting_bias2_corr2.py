"""Create Bias2 correlation plots focusing on stain and class metrics.

This module reads the detailed Bias2 correlation evaluation CSV and produces
multi-page PDF reports summarising how stain-level and class-level metrics vary
with the ``Scale`` parameter.  A single figure aggregates all models together
followed by one figure per individual model.  Each figure contains a grid of
plots with stain metrics on the left and class metrics on the right.

The PDF is stored in ``analysis/plots/bias2_corr_plots2.pdf``.

Usage
-----
Run the script directly:

```
python -m analysis.plotting_bias2_corr2
```

"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

DATA_PATH = Path("analysis/tables/bias2_corr_model_evaluation_detailed.csv")
OUTPUT_PATH = Path("analysis/plots/bias2_corr_plots2.pdf")

MODEL_COLUMN = "Model"
SCALE_COLUMN = "Scale"
FOLD_COLUMN = "Fold"

METRICS: Sequence[str] = ("Accuracy", "AUC", "F1", "Precision")
FIGURE_ROWS = max(5, len(METRICS) + 1)
FIGURE_COLS = 2
FIGURE_SIZE = (14, FIGURE_ROWS * 2.6)


def load_bias2_corr_results(csv_path: Path) -> pd.DataFrame:
    """Load the Bias2 correlation CSV file and ensure numeric values."""

    if not csv_path.exists():  # pragma: no cover - runtime validation
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in df.columns:
        if column == MODEL_COLUMN:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if SCALE_COLUMN in df.columns:
        df[SCALE_COLUMN] = pd.to_numeric(df[SCALE_COLUMN], errors="coerce")

    if FOLD_COLUMN in df.columns:
        df[FOLD_COLUMN] = df[FOLD_COLUMN].astype(pd.Int64Dtype())

    df = df.dropna(subset=[SCALE_COLUMN])
    return df


def ensure_output_directory(output_path: Path) -> None:
    """Create the output directory tree if necessary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _compute_metric_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return mean and deviation of ``column`` grouped by ``Scale``."""

    if column not in df.columns:
        return pd.DataFrame()

    subset = df[[SCALE_COLUMN, column]].dropna()
    if subset.empty:
        return pd.DataFrame()

    grouped = subset.groupby(SCALE_COLUMN)[column]
    summary = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(ddof=0),
            "count": grouped.count(),
        }
    ).sort_index()

    summary["std"].fillna(0.0, inplace=True)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    summary["lower"] = summary["lower"].clip(lower=0.0, upper=1.0)
    summary["upper"] = summary["upper"].clip(lower=0.0, upper=1.0)
    return summary


def _compute_average_summary(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Compute a summary using the row-wise average of ``columns``."""

    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.DataFrame()

    subset = df[[SCALE_COLUMN] + available].copy()
    subset["_average_metric"] = subset[available].mean(axis=1, skipna=True)
    subset = subset[[SCALE_COLUMN, "_average_metric"]].dropna()

    grouped = subset.groupby(SCALE_COLUMN)["_average_metric"]
    summary = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(ddof=0),
            "count": grouped.count(),
        }
    ).sort_index()

    summary["std"].fillna(0.0, inplace=True)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    summary["lower"] = summary["lower"].clip(lower=0.0, upper=1.0)
    summary["upper"] = summary["upper"].clip(lower=0.0, upper=1.0)
    return summary


def _format_metric_label(prefix: str, metric: str) -> str:
    return f"{prefix} {metric}".strip()


def _format_average_label(prefix: str) -> str:
    metric_list = ", ".join(METRICS)
    return f"{prefix} Average ({metric_list})"


def _plot_metric_line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    """Render a line plot for the provided summary statistics."""

    if summary.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    scales = summary.index.to_numpy(dtype=float)
    means = summary["mean"].to_numpy(dtype=float)
    errors = summary["std"].to_numpy(dtype=float)

    (line,) = ax.plot(scales, means, marker="o", linewidth=2)
    if np.any(errors > 0):
        lower = summary["lower"].to_numpy(dtype=float)
        upper = summary["upper"].to_numpy(dtype=float)
        ax.fill_between(scales, lower, upper, color=line.get_color(), alpha=0.2)

    ax.set_title(title)

    if show_xlabel:
        ax.set_xlabel("Scale")
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("Metric Value")
    else:
        ax.set_ylabel("")

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(scales)
    ax.set_xticklabels([f"{value:g}" for value in scales])
    ax.grid(True, linestyle="--", alpha=0.3)


def _populate_metric_grid(axes: np.ndarray, df: pd.DataFrame) -> None:
    """Fill the axes grid with stain and class metric plots."""

    num_rows = axes.shape[0]
    for row_idx, metric in enumerate(METRICS):
        if row_idx >= num_rows:
            break

        stain_column = f"stain_{metric}"
        class_column = f"class_{metric}"

        stain_summary = _compute_metric_summary(df, stain_column)
        class_summary = _compute_metric_summary(df, class_column)

        show_xlabel = row_idx == num_rows - 1
        _plot_metric_line(
            axes[row_idx, 0],
            stain_summary,
            _format_metric_label("Stain", metric),
            show_xlabel=show_xlabel,
            show_ylabel=True,
        )
        _plot_metric_line(
            axes[row_idx, 1],
            class_summary,
            _format_metric_label("Class", metric),
            show_xlabel=show_xlabel,
            show_ylabel=False,
        )

    if num_rows > len(METRICS):
        average_row = num_rows - 1
        stain_columns = [f"stain_{metric}" for metric in METRICS]
        class_columns = [f"class_{metric}" for metric in METRICS]

        stain_summary = _compute_average_summary(df, stain_columns)
        class_summary = _compute_average_summary(df, class_columns)

        _plot_metric_line(
            axes[average_row, 0],
            stain_summary,
            _format_average_label("Stain"),
            show_xlabel=True,
            show_ylabel=True,
        )
        _plot_metric_line(
            axes[average_row, 1],
            class_summary,
            _format_average_label("Class"),
            show_xlabel=True,
            show_ylabel=False,
        )


def _render_metric_figure(pdf: PdfPages, df: pd.DataFrame, title: str) -> None:
    """Create a figure with stain and class metric trends."""

    fig, axes = plt.subplots(
        FIGURE_ROWS,
        FIGURE_COLS,
        figsize=FIGURE_SIZE,
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    _populate_metric_grid(axes, df)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _iterate_models(df: pd.DataFrame) -> Iterable[str]:
    """Yield model names in alphabetical order."""

    if MODEL_COLUMN not in df.columns:
        return []

    models = df[MODEL_COLUMN].dropna().astype(str).unique()
    return sorted(models)


def main() -> None:
    df = load_bias2_corr_results(DATA_PATH)
    ensure_output_directory(OUTPUT_PATH)

    with PdfPages(OUTPUT_PATH) as pdf:
        _render_metric_figure(pdf, df, "All Models (Combined)")

        for model in _iterate_models(df):
            model_df = df[df[MODEL_COLUMN] == model]
            _render_metric_figure(pdf, model_df, f"Model: {model}")

    print(f"Saved plots to {OUTPUT_PATH}")


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
