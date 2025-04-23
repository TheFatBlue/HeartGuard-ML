# src/utils/fairness.py
"""
Functions for analyzing model fairness across demographic slices. [cite: 18]
"""
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import logging
from src.utils.metrics import evaluate_classifier # Reuse evaluation function

logger = logging.getLogger(__name__)

def compute_slice_metrics(X_test, y_true, y_pred, y_proba, slice_features):
    """
    Compute performance metrics for different demographic slices.

    Parameters:
    -----------
    X_test : pd.DataFrame
        Test features, including slice features.
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    y_proba : array-like
        Predicted probabilities.
    slice_features : list
        List of column names in X_test to slice by.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing metrics for each slice.
    """
    logger.info(f"Computing fairness metrics across slices: {slice_features}")
    all_slice_metrics = []

    # Ensure y_true, y_pred, y_proba are pd.Series with same index as X_test
    if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true, index=X_test.index)
    if not isinstance(y_pred, pd.Series): y_pred = pd.Series(y_pred, index=X_test.index)
    if y_proba is not None and not isinstance(y_proba, pd.Series): y_proba = pd.Series(y_proba, index=X_test.index)


    for feature in slice_features:
        if feature not in X_test.columns:
            logger.warning(f"Slice feature '{feature}' not found in X_test. Skipping.")
            continue

        logger.info(f"--- Analyzing slices for feature: {feature} ---")
        unique_slices = X_test[feature].unique()

        for slice_value in unique_slices:
            if pd.isna(slice_value): # Handle potential NaN slice values
                 slice_mask = X_test[feature].isna()
                 slice_name = f"{feature}_NaN"
            else:
                 slice_mask = (X_test[feature] == slice_value)
                 slice_name = f"{feature}_{slice_value}"

            slice_y_true = y_true[slice_mask]
            slice_y_pred = y_pred[slice_mask]
            slice_y_proba = y_proba[slice_mask] if y_proba is not None else None

            if len(slice_y_true) == 0:
                logger.warning(f"Slice '{slice_name}' is empty. Skipping.")
                continue
            if len(np.unique(slice_y_true)) < 2:
                logger.warning(f"Slice '{slice_name}' has only one class ({np.unique(slice_y_true)}). Metrics might be undefined.")
                # Handle single-class slices gracefully in evaluate_classifier if needed
                # For now, we rely on zero_division='warn' or 0 in scikit-learn metrics

            logger.debug(f"Evaluating slice: {slice_name} (Size: {len(slice_y_true)})")
            try:
                slice_metrics = evaluate_classifier(slice_y_true, slice_y_pred, slice_y_proba)
                slice_metrics['slice_feature'] = feature
                slice_metrics['slice_value'] = slice_value
                slice_metrics['slice_name'] = slice_name
                slice_metrics['slice_size'] = len(slice_y_true)
                all_slice_metrics.append(slice_metrics)
            except Exception as e:
                logger.error(f"Error evaluating slice '{slice_name}': {e}")


    if not all_slice_metrics:
         logger.warning("No slice metrics were computed.")
         return pd.DataFrame()

    metrics_df = pd.DataFrame(all_slice_metrics)

    # Reorder columns for readability
    cols = ['slice_feature', 'slice_value', 'slice_name', 'slice_size', 'recall', 'precision', 'f1', 'accuracy', 'f_beta', 'roc_auc', 'pr_auc', 'false_negatives', 'false_positives']
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]] # Keep only existing columns

    logger.info(f"Successfully computed metrics for {len(metrics_df)} slices.")
    return metrics_df

def export_fairness_report(metrics_df, output_path, format='csv'):
    """
    Export the fairness metrics report to CSV or Markdown.

    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with slice metrics.
    output_path : Path object or str
        File path to save the report.
    format : str, default='csv'
        Output format ('csv' or 'markdown').
    """
    logger.info(f"Exporting fairness report to {output_path} in {format} format")
    try:
        if format == 'csv':
            metrics_df.to_csv(output_path, index=False)
        elif format == 'markdown':
            metrics_df.to_markdown(output_path, index=False)
        else:
            raise ValueError("Unsupported format. Choose 'csv' or 'markdown'.")
        logger.info("Fairness report exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export fairness report: {e}")
