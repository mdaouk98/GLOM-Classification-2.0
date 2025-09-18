import json
import logging
import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from main_utils.config import load_config
from main_utils.train_utils import DataLoadingError, load_hdf5_data, split_data


DEFAULT_CONFIG_PATH = \
    "configs/finale3/image_size_224/model_Resnet18.yaml"
BASE_METRICS_DIR = "metrics/bias2"
OUTPUT_DETAILED_PATH = \
    "analysis/tables/bias2_model_evaluation_detailed.csv"

label = "stain"  # Change to "scanner" to analyze scanner bias
BIAS_HEAD = 'stain' # "stain" or "scanner" or None to auto-detect
MODELS = [
    "Resnet18",
    "Resnet34",
    "Resnet50",
    "Resnet101",
    "Resnet152",
    "Densenet121",
    "Densenet169",
    "Densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "regnety_008",
    "regnety_016",
    "resnext101_32x8d",
    "Vision",
]

FOLDS = range(1, 6)



logger = logging.getLogger(__name__)


def compute_uncertainty(softmax_array: np.ndarray) -> float:
    """Calculate predictive entropy (uncertainty)."""

    mean_softmax = np.mean(softmax_array, axis=0)  # shape: (N, C)
    return float(np.mean(entropy(mean_softmax.T)))


def load_dataset(config_path: str) -> Dict[str, np.ndarray]:
    """Load dataset information required for evaluation."""

    config = load_config(config_path)
    try:
        (
            all_indices,
            all_labels,
            all_wsis,
            all_stain,
            all_scanner,
            total_samples,
        ) = load_hdf5_data(config)
        logger.info("Total samples loaded: %s", total_samples)
        if isinstance(all_labels, dict):
            logger.info(
                "Available label heads: %s",
                ", ".join(sorted(all_labels.keys())),
            )
        else:
            labels_array = np.asarray(all_labels)
            logger.debug(
                "All labels shape: %s, unique labels: %s",
                labels_array.shape,
                np.unique(labels_array),
            )
        stain_array = np.asarray(all_stain)
        logger.debug(
            "All stains shape: %s, unique stains: %s",
            stain_array.shape,
            np.unique(stain_array),
        )
        scanner_array = np.asarray(all_scanner)
        logger.debug(
            "All scanners shape: %s, unique scanners: %s",
            scanner_array.shape,
            np.unique(scanner_array),
        )
    except DataLoadingError as exc:  # pragma: no cover - logging for runtime visibility
        logging.error("[Data Loading] %s", exc)
        raise

    _, _, test_idx, _ = split_data(
        all_indices, all_labels, all_wsis, config
    )
    test_idx = np.array(test_idx)

    if isinstance(all_labels, dict):
        labels_by_head = {
            head: np.asarray(values)[test_idx]
            for head, values in all_labels.items()
        }
    else:
        labels_by_head = {"class": np.asarray(all_labels)[test_idx]}

    test_class_labels = labels_by_head.get("class")
    if test_class_labels is not None:
        logger.debug(
            "Test class labels shape: %s, unique labels: %s",
            test_class_labels.shape,
            np.unique(test_class_labels),
        )

    test_stains = np.array(all_stain)[test_idx]
    unique_test_stains = np.unique(test_stains)
    logger.info(
        "Test stain distribution: %s",
        {stain: np.sum(test_stains == stain) for stain in unique_test_stains},
    )

    test_scanners = np.array(all_scanner)[test_idx]
    unique_test_scanners = np.unique(test_scanners)
    logger.info(
        "Test scanner distribution: %s",
        {
            scanner: np.sum(test_scanners == scanner)
            for scanner in unique_test_scanners
        },
    )

    return {
        "config": config,
        "test_labels": test_class_labels,
        "labels_by_head": labels_by_head,
        "test_indices": test_idx,
        "test_stains": test_stains,
        "unique_test_stains": unique_test_stains,
        "test_scanners": test_scanners,
        "unique_test_scanners": unique_test_scanners,
        "all_labels": all_labels,
        "all_stain": all_stain,
        "all_scanner": all_scanner,
        "all_wsis": all_wsis,
    }


def build_metric_path(
    base_dir: str, model: str, fold: int
) -> str:
    """Construct the file path for a specific metric JSON file."""

    return os.path.join(
        base_dir,
        f"{label}",
        f"model_{model}",
        f"training_dict_fold{fold}.json",
    )


def load_metric_file(metric_path: str) -> Optional[Dict]:
    """Load metrics from disk, handling missing files and JSON errors."""

    if not os.path.exists(metric_path):
        logger.warning("File not found: %s. Skipping.", metric_path)
        return None

    try:
        with open(metric_path, "r", encoding="utf-8") as metric_file:
            return json.load(metric_file)
    except json.decoder.JSONDecodeError as exc:
        logger.error("Error decoding JSON in %s: %s", metric_path, exc)
    return None


def compute_auc(labels: np.ndarray, mean_softmax: np.ndarray) -> float:
    """Compute ROC AUC for binary or multiclass predictions."""

    n_classes = mean_softmax.shape[1]
    unique_labels = np.unique(labels)

    try:
        if n_classes == 2 and len(unique_labels) == 2:
            return float(roc_auc_score(labels, mean_softmax[:, 1]))
        lb = label_binarize(labels, classes=np.arange(n_classes))
        return float(
            roc_auc_score(lb, mean_softmax, multi_class="ovr", average="macro")
        )
    except ValueError as exc:
        logger.warning("Error computing AUC: %s", exc)
        return float("nan")


def compute_per_bias_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    mean_softmax: np.ndarray,
    softmax_iters: np.ndarray,
    bias_values: np.ndarray,
    unique_bias_values: np.ndarray,
    prefix: str,
    n_classes: int,
) -> Dict[str, float]:
    """Compute metrics for each bias category (stain, scanner, ...)."""

    per_bias_metrics: Dict[str, float] = {}

    if len(bias_values) != labels.shape[0]:
        logger.warning(
            "Skipping per-%s metrics: mismatch between %s labels (%d) and predictions (%d).",
            prefix.lower(),
            prefix.lower(),
            len(bias_values),
            labels.shape[0],
        )
        return per_bias_metrics

    for bias_value in unique_bias_values:
        bias_mask = bias_values == bias_value
        if not np.any(bias_mask):
            continue

        labels_subset = labels[bias_mask]
        preds_subset = preds[bias_mask]
        softmax_subset = mean_softmax[bias_mask]
        softmax_iters_subset = softmax_iters[:, bias_mask, :]

        bias_accuracy = accuracy_score(labels_subset, preds_subset)
        bias_precision = precision_score(
            labels_subset, preds_subset, average="macro", zero_division=0
        )
        bias_recall = recall_score(
            labels_subset, preds_subset, average="macro", zero_division=0
        )
        bias_f1 = f1_score(
            labels_subset, preds_subset, average="macro", zero_division=0
        )

        try:
            unique_subset_labels = np.unique(labels_subset)
            if n_classes == 2 and len(unique_subset_labels) == 2:
                bias_auc = roc_auc_score(labels_subset, softmax_subset[:, 1])
            else:
                lb_subset = label_binarize(
                    labels_subset, classes=np.arange(n_classes)
                )
                bias_auc = roc_auc_score(
                    lb_subset, softmax_subset, multi_class="ovr", average="macro"
                )
        except ValueError:
            bias_auc = np.nan

        bias_uncertainty = compute_uncertainty(softmax_iters_subset)

        cm_subset = confusion_matrix(
            labels_subset, preds_subset, labels=np.arange(n_classes)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            bias_class_acc = np.divide(
                cm_subset.diagonal().astype(float),
                cm_subset.sum(axis=1).astype(float),
                out=np.full(n_classes, np.nan),
                where=cm_subset.sum(axis=1) != 0,
            )

        per_bias_metrics.update(
            {
                f"Per{prefix}_{bias_value}_Accuracy": bias_accuracy,
                f"Per{prefix}_{bias_value}_Precision": bias_precision,
                f"Per{prefix}_{bias_value}_Recall": bias_recall,
                f"Per{prefix}_{bias_value}_F1": bias_f1,
                f"Per{prefix}_{bias_value}_AUC": bias_auc,
                f"Per{prefix}_{bias_value}_Uncertainty": bias_uncertainty,
            }
        )

        for class_index, class_value in enumerate(bias_class_acc):
            per_bias_metrics[
                f"Per{prefix}_{bias_value}_Acc_class_{class_index}"
            ] = class_value

    return per_bias_metrics


def compute_per_stain_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    mean_softmax: np.ndarray,
    softmax_iters: np.ndarray,
    stain_info: Dict[str, np.ndarray],
    n_classes: int,
) -> Dict[str, float]:
    """Compute per-stain metrics."""

    return compute_per_bias_metrics(
        labels,
        preds,
        mean_softmax,
        softmax_iters,
        stain_info["test_stains"],
        stain_info["unique_test_stains"],
        "Stain",
        n_classes,
    )


def compute_per_scanner_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    mean_softmax: np.ndarray,
    softmax_iters: np.ndarray,
    scanner_info: Dict[str, np.ndarray],
    n_classes: int,
) -> Dict[str, float]:
    """Compute per-scanner metrics."""

    return compute_per_bias_metrics(
        labels,
        preds,
        mean_softmax,
        softmax_iters,
        scanner_info["test_scanners"],
        scanner_info["unique_test_scanners"],
        "Scanner",
        n_classes,
    )


def compute_classification_metrics(
    base_info: Dict[str, float],
    labels: np.ndarray,
    softmax_iters: np.ndarray,
    stain_info: Optional[Dict[str, np.ndarray]] = None,
    scanner_info: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute metrics for a given set of logits."""

    mean_softmax = np.mean(softmax_iters, axis=0)
    preds = np.argmax(mean_softmax, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    auc = compute_auc(labels, mean_softmax)
    uncertainty = compute_uncertainty(softmax_iters)

    n_classes = mean_softmax.shape[1]
    cm = confusion_matrix(labels, preds, labels=np.arange(n_classes))
    with np.errstate(divide="ignore", invalid="ignore"):
        class_acc = np.divide(
            cm.diagonal().astype(float),
            cm.sum(axis=1).astype(float),
            out=np.full(n_classes, np.nan),
            where=cm.sum(axis=1) != 0,
        )

    per_stain_metrics: Dict[str, float] = {}
    if stain_info is not None:
        per_stain_metrics = compute_per_stain_metrics(
            labels, preds, mean_softmax, softmax_iters, stain_info, n_classes
        )

    per_scanner_metrics: Dict[str, float] = {}
    if scanner_info is not None:
        per_scanner_metrics = compute_per_scanner_metrics(
            labels, preds, mean_softmax, softmax_iters, scanner_info, n_classes
        )

    return {
        **base_info,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "Uncertainty": uncertainty,
        **{f"Acc_class_{i}": class_acc[i] for i in range(n_classes)},
        **per_stain_metrics,
        **per_scanner_metrics,
    }


def add_metrics(
    dict_metrics: Dict[str, Dict[str, float]],
    key_list: Iterable[str],
    base_entry: Dict[str, float],
    num_epochs_trained: Optional[int],
    metrics_list: List[Dict[str, float]],
    detailed_metrics: List[Dict[str, float]],
) -> None:
    """Store computed metrics for later aggregation and export."""

    entry = dict(base_entry)
    for key in key_list:
        for metric_name, metric_value in dict_metrics[key].items():
            if metric_name in ("Model", "Fold"):
                continue
            entry[f"{key}_{metric_name}"] = metric_value

    metrics_list.append(entry)
    entry["num_epochs_trained"] = num_epochs_trained
    detailed_metrics.append(entry)


def compute_summary(
    metrics_list: List[Dict[str, float]],
    model: str,
) -> Optional[Dict[str, float]]:
    """Compute aggregate statistics across collected metrics."""

    if not metrics_list:
        return None

    df = pd.DataFrame(metrics_list)

    def series(column: str) -> pd.Series:
        return df.get(column, pd.Series(dtype=float))

    if BIAS_HEAD is None:
        bias_prefix = BIAS_HEAD
        bias_title = BIAS_HEAD
    else:
        bias_prefix = BIAS_HEAD
        bias_title = BIAS_HEAD.capitalize()

    return {
        "Model": model,
        "Class_Accuracy_mean": series("class_Accuracy").mean(),
        "Class_Accuracy_std": series("class_Accuracy").std(),
        "Class_Accuracy_var": series("class_Accuracy").var(),
        "Class_Precision_mean": series("class_Precision").mean(),
        "Class_Precision_std": series("class_Precision").std(),
        "Class_Precision_var": series("class_Precision").var(),
        "Class_Recall_mean": series("class_Recall").mean(),
        "Class_Recall_std": series("class_Recall").std(),
        "Class_Recall_var": series("class_Recall").var(),
        "Class_F1_mean": series("class_F1").mean(),
        "Class_F1_std": series("class_F1").std(),
        "Class_F1_var": series("class_F1").var(),
        "Class_AUC_mean": series("class_AUC").mean(),
        "Class_AUC_std": series("class_AUC").std(),
        "Class_AUC_var": series("class_AUC").var(),
        "Class_Uncertainty_mean": series("class_Uncertainty").mean(),
        "Class_Uncertainty_std": series("class_Uncertainty").std(),
        "Class_Uncertainty_var": series("class_Uncertainty").var(),
        "Class_Acc_class_0_mean": series("class_Acc_class_0").mean(),
        "Class_Acc_class_1_mean": series("class_Acc_class_1").mean(),
        f"{bias_title}_Accuracy_mean": series(f"{bias_prefix}_Accuracy").mean(),
        f"{bias_title}_Accuracy_std": series(f"{bias_prefix}_Accuracy").std(),
        f"{bias_title}_Accuracy_var": series(f"{bias_prefix}_Accuracy").var(),
        f"{bias_title}_Precision_mean": series(f"{bias_prefix}_Precision").mean(),
        f"{bias_title}_Precision_std": series(f"{bias_prefix}_Precision").std(),
        f"{bias_title}_Precision_var": series(f"{bias_prefix}_Precision").var(),
        f"{bias_title}_Recall_mean": series(f"{bias_prefix}_Recall").mean(),
        f"{bias_title}_Recall_std": series(f"{bias_prefix}_Recall").std(),
        f"{bias_title}_Recall_var": series(f"{bias_prefix}_Recall").var(),
        f"{bias_title}_F1_mean": series(f"{bias_prefix}_F1").mean(),
        f"{bias_title}_F1_std": series(f"{bias_prefix}_F1").std(),
        f"{bias_title}_F1_var": series(f"{bias_prefix}_F1").var(),
        f"{bias_title}_AUC_mean": series(f"{bias_prefix}_AUC").mean(),
        f"{bias_title}_AUC_std": series(f"{bias_prefix}_AUC").std(),
        f"{bias_title}_AUC_var": series(f"{bias_prefix}_AUC").var(),
        f"{bias_title}_Uncertainty_mean": series(f"{bias_prefix}_Uncertainty").mean(),
        f"{bias_title}_Uncertainty_std": series(f"{bias_prefix}_Uncertainty").std(),
        f"{bias_title}_Uncertainty_var": series(f"{bias_prefix}_Uncertainty").var(),
        f"{bias_title}_Acc_class_0_mean": series(f"{bias_prefix}_Acc_class_0").mean(),
        f"{bias_title}_Acc_class_1_mean": series(f"{bias_prefix}_Acc_class_1").mean(),
    }


def process_metric_data(
    model: str,
    fold: int,
    data: Dict,
    dataset_info: Dict[str, np.ndarray],
    metrics_list: List[Dict[str, float]],
    detailed_metrics: List[Dict[str, float]],
    results: List[Dict[str, float]],
) -> None:
    """Process a single metric JSON structure."""

    testing_dict = data.get("all_testing_dict", {})
    logits_dict = testing_dict.get("all_logits", {})
    labels_dict = testing_dict.get("all_labels", {})

    dict_metrics: Dict[str, Dict[str, float]] = {}
    key_list: List[str] = []

    base_info = {
        "Model": model,
        "Fold": fold,
    }

    labels_by_head: Dict[str, np.ndarray] = dataset_info.get("labels_by_head", {})

    # Handle case where logits_dict is a list (single head) instead of dict (multi head)
    if isinstance(logits_dict, list):
        logits_dict = {f"{label}": logits_dict}
        if isinstance(labels_dict, list):
            labels_dict = {f"{label}": labels_dict}

    for key, logits in logits_dict.items():
        logger.info(
            "Processing head '%s' for model %s, fold %d",
            key,
            model,
            fold,
        )
        key_list.append(key)

        logits_iters = np.array(logits)
        logger.debug("logits_iters shape: %s", logits_iters.shape)
        softmax_iters = softmax(logits_iters, axis=2)

        stain_info: Optional[Dict[str, np.ndarray]] = None
        scanner_info: Optional[Dict[str, np.ndarray]] = None

        if key == "class":
            label_source = labels_by_head.get("class")
            if label_source is None and key in labels_dict:
                label_source = labels_dict.get(key)
            if label_source is None:
                logger.warning("No labels available for head '%s'. Skipping.", key)
                continue

            labels = np.asarray(label_source)
            stain_info = {
                "test_stains": dataset_info["test_stains"],
                "unique_test_stains": dataset_info["unique_test_stains"],
            }
            scanner_info = {
                "test_scanners": dataset_info["test_scanners"],
                "unique_test_scanners": dataset_info["unique_test_scanners"],
            }
        elif key == BIAS_HEAD:
            if BIAS_HEAD == "scanner":
                labels = np.asarray(dataset_info["test_scanners"])
            elif BIAS_HEAD == "stain":
                labels = np.asarray(dataset_info["test_stains"])
            else:
                bias_labels = labels_by_head.get(key)
                if bias_labels is None and key in labels_dict:
                    bias_labels = labels_dict.get(key)
                if bias_labels is None:
                    logger.warning(
                        "No labels available for bias head '%s'. Skipping.", key
                    )
                    continue
                labels = np.asarray(bias_labels)
        else:
            label_source = labels_by_head.get(key)
            if label_source is None and key in labels_dict:
                label_source = labels_dict.get(key)
            if label_source is None:
                logger.warning("No labels available for head '%s'. Skipping.", key)
                continue
            labels = np.asarray(label_source)

        logger.debug(
            "labels shape: %s, unique labels: %s",
            labels.shape,
            np.unique(labels),
        )

        dict_metrics[key] = compute_classification_metrics(
            base_info,
            labels,
            softmax_iters,
            stain_info=stain_info,
            scanner_info=scanner_info,
        )

    add_metrics(
        dict_metrics,
        key_list,
        base_info,
        data.get("num_epochs_trained"),
        metrics_list,
        detailed_metrics,
    )

    summary = compute_summary(metrics_list, model)
    if summary is not None:
        results.append(summary)


def process_model(
    model: str,
    dataset_info: Dict[str, np.ndarray],
    metrics_list: List[Dict[str, float]],
    detailed_metrics: List[Dict[str, float]],
    results: List[Dict[str, float]],
    base_dir: str,
    folds: Iterable[int],
) -> None:
    """Process all combinations of hyperparameters for a given model."""

    logger.info("Processing model %s", model)
    for fold in folds:
        metric_path = build_metric_path(
            base_dir, model, fold
        )
        data = load_metric_file(metric_path)
        if data is None:
            continue
        process_metric_data(
            model,
            fold,
            data,
            dataset_info,
            metrics_list,
            detailed_metrics,
            results,
        )


def save_detailed_metrics(
    detailed_metrics: List[Dict[str, float]], output_path: str
) -> None:
    """Save detailed metrics to disk."""

    detailed_df = pd.DataFrame(detailed_metrics)
    detailed_df.to_csv(output_path, index=False)


def main(
    config_path: str = DEFAULT_CONFIG_PATH,
    base_metrics_dir: str = BASE_METRICS_DIR,
    output_path: str = OUTPUT_DETAILED_PATH,
    models: Iterable[str] = MODELS,
    folds: Iterable[int] = FOLDS,
) -> None:
    """Entry point for generating the multihead evaluation tables."""

    logging.basicConfig(level=logging.INFO)
    logger.info("Loading dataset using config at %s", config_path)
    dataset_info = load_dataset(config_path)

    metrics_list: List[Dict[str, float]] = []
    detailed_metrics: List[Dict[str, float]] = []
    results: List[Dict[str, float]] = []

    for model in models:
        process_model(
            model,
            dataset_info,
            metrics_list,
            detailed_metrics,
            results,
            base_metrics_dir,
            folds,
        )

    if detailed_metrics:
        logger.info("Saving detailed metrics to %s", output_path)
        save_detailed_metrics(detailed_metrics, output_path)


if __name__ == "__main__":
    main()
