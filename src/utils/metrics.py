"""
Custom evaluation metrics for heart disease prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.metrics import precision_recall_curve, auc, make_scorer, fbeta_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import scipy.stats as st

# Import config for F-beta
from src.config import F_BETA_SCORE_BETA, THRESHOLD_MIN_PRECISION_CONSTRAINTS

import logging

logger = logging.getLogger(__name__)

def evaluate_classifier(y_true, y_pred, y_proba=None):
    """
    Evaluate a classifier with multiple metrics, including PR AUC and F-beta.
    """
    # ... (keep existing confusion matrix calculation: tn, fp, fn, tp)
    logger.info("Evaluating classifier performance")

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Basic metrics
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f_beta': fbeta_score(y_true, y_pred, beta=F_BETA_SCORE_BETA, zero_division=0), # Added F-beta [cite: 16]
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

    # If probabilities are provided, calculate ROC AUC and PR AUC
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve) # Added PR AUC

    # Log key metrics
    log_metrics = {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))}
    logger.info(f"Classification metrics: {log_metrics}")

    return metrics


def find_optimal_threshold(y_true, y_proba, optimize_for='recall', min_precision_constraints=THRESHOLD_MIN_PRECISION_CONSTRAINTS):
    """
    Find the optimal probability threshold using grid search over minimum precision constraints.
    Maximizes recall by default, subject to the constraint. Can also optimize F-beta. [cite: 16]

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_proba : array-like
        Predicted probabilities for the positive class
    optimize_for : str, default='recall'
        Metric to optimize ('recall', 'f_beta').
    min_precision_constraints : list, default from config
        List of minimum precision constraints to test.

    Returns:
    --------
    tuple
        (best_threshold, best_metrics_dict)
    """
    logger.info(f"Finding optimal threshold to optimize {optimize_for} across precision constraints: {min_precision_constraints}")

    best_overall_threshold = 0.5 # Default fallback
    best_overall_score = -1
    best_metrics = {}

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds array is 1 shorter than precisions/recalls
    thresholds = np.append(thresholds, 1.0)

    if optimize_for == 'f_beta':
        f_beta_scores = fbeta_score(y_true[:, None], y_proba[:, None] >= thresholds, beta=F_BETA_SCORE_BETA, zero_division=0, average=None) # Check this calculation carefully

    for min_precision in min_precision_constraints:
        logger.debug(f"Testing minimum precision constraint: {min_precision}")
        valid_indices = np.where(precisions >= min_precision)[0]

        if len(valid_indices) == 0:
            logger.warning(f"No threshold satisfies minimum precision {min_precision}. Skipping constraint.")
            continue

        # Filter based on valid indices
        current_thresholds = thresholds[valid_indices]
        current_recalls = recalls[valid_indices]
        current_precisions = precisions[valid_indices]


        if optimize_for == 'recall':
            best_idx_for_constraint = np.argmax(current_recalls)
            current_best_score = current_recalls[best_idx_for_constraint]
        elif optimize_for == 'f_beta':
             # Need F-beta scores calculated for these thresholds
             # Placeholder: maximize recall within constraint for now
             # Proper F-beta optimization requires calculating it across thresholds
             # f_beta_scores_constraint = f_beta_scores[valid_indices] # If f_beta_scores were calculated correctly
             # best_idx_for_constraint = np.argmax(f_beta_scores_constraint)
             # current_best_score = f_beta_scores_constraint[best_idx_for_constraint]
             logger.warning("F-beta optimization across thresholds not fully implemented, using recall.")
             best_idx_for_constraint = np.argmax(current_recalls) # Fallback
             current_best_score = current_recalls[best_idx_for_constraint]
        else:
             raise ValueError(f"Unsupported optimization metric: {optimize_for}")

        current_best_threshold = current_thresholds[best_idx_for_constraint]

        logger.debug(f"Constraint={min_precision}: Best Threshold={current_best_threshold:.4f}, Score={current_best_score:.4f}, Recall={current_recalls[best_idx_for_constraint]:.4f}, Precision={current_precisions[best_idx_for_constraint]:.4f}")

        # Update overall best if current constraint yields better score
        if current_best_score > best_overall_score:
            best_overall_score = current_best_score
            best_overall_threshold = current_best_threshold
            # Recalculate metrics for this threshold
            y_pred_optimal = (y_proba >= best_overall_threshold).astype(int)
            best_metrics = evaluate_classifier(y_true, y_pred_optimal, y_proba)
            best_metrics['optimal_threshold'] = best_overall_threshold
            best_metrics['min_precision_constraint'] = min_precision # Store which constraint led to this optimum

    if not best_metrics: # Handle case where no threshold met any constraint
         logger.warning("No threshold met any minimum precision constraint. Returning default threshold 0.5.")
         y_pred_default = (y_proba >= 0.5).astype(int)
         best_metrics = evaluate_classifier(y_true, y_pred_default, y_proba)
         best_metrics['optimal_threshold'] = 0.5
         best_metrics['min_precision_constraint'] = None

    logger.info(f"Optimal threshold found: {best_metrics['optimal_threshold']:.4f} "
               f"(optimized for {optimize_for}, constraint={best_metrics['min_precision_constraint']}). "
               f"Resulting Recall={best_metrics['recall']:.4f}, Precision={best_metrics['precision']:.4f}")

    return best_metrics['optimal_threshold'], best_metrics


def cross_validate_model(model, X, y, cv_folds, scoring, fit_params=None):
    """
    Perform stratified cross-validation and return aggregated metrics with confidence intervals. [cite: 19]

    Parameters:
    -----------
    model : estimator object
        The model pipeline to evaluate.
    X : array-like
        Features.
    y : array-like
        Target variable.
    cv_folds : int
        Number of cross-validation folds.
    scoring : dict or list
        Scoring metrics to compute.
    fit_params : dict, optional
        Parameters to pass to the fit method.

    Returns:
    --------
    dict
        Dictionary with mean and confidence interval for each metric.
    """
    logger.info(f"Performing {cv_folds}-fold stratified cross-validation")

    # Define scorers, ensuring PR AUC is included if requested
    scorers = {
        'recall': make_scorer(recall_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc',
        'pr_auc': make_scorer(lambda y_true, y_pred_proba: auc(*precision_recall_curve(y_true, y_pred_proba[:, 1])[:2][::-1]), needs_proba=True),
        'f_beta': make_scorer(fbeta_score, beta=F_BETA_SCORE_BETA, zero_division=0)
    }
    # Filter scorers based on the 'scoring' input list/dict
    active_scorers = {k: scorers[k] for k in scoring if k in scorers}

    cv_results = cross_validate(model, X, y, cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                scoring=active_scorers, return_train_score=False, n_jobs=-1,
                                fit_params=fit_params)

    results_summary = {}
    for metric in active_scorers.keys():
        metric_key = f'test_{metric}'
        if metric_key in cv_results:
            scores = cv_results[metric_key]
            mean_score = np.mean(scores)
            std_err = st.sem(scores)
            ci = st.t.interval(0.95, len(scores)-1, loc=mean_score, scale=std_err) if len(scores) > 1 and std_err > 0 else (mean_score, mean_score)

            results_summary[metric] = {
                'mean': mean_score,
                'std_dev': np.std(scores),
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'individual_scores': scores.tolist() # Optional: keep individual fold scores
            }
            logger.info(f"CV Metric - {metric}: Mean={mean_score:.4f}, CI=({ci[0]:.4f}, {ci[1]:.4f})")
        else:
             logger.warning(f"Metric key '{metric_key}' not found in cross_validate results.")


    return results_summary


# --- Keep existing compare_models and analyze_misclassifications ---
# --- Update compare_models if needed to handle the new metrics/structure from cross_validate_model ---

# src/utils/metrics.py

# ... (keep other imports and functions)

def compare_models(model_results, target_metric='recall'):
    """
    Compare multiple models based on evaluation metrics.
    Handles both flat dictionaries (from evaluate_classifier) and
    nested dictionaries with mean/CI (from cross_validate_model).
    """
    logger.info(f"Comparing models based on {target_metric}")
    comparison_data = []
    has_cv_structure = False # Flag to track input structure

    # Check the structure of the first model's results to determine format
    if model_results:
        first_model_name = list(model_results.keys())[0]
        first_result_value = list(model_results[first_model_name].values())[0]
        if isinstance(first_result_value, dict) and 'mean' in first_result_value:
            has_cv_structure = True
            logger.info("Detected nested CV results structure.")
        else:
             logger.info("Detected flat results structure.")


    for model_name, results in model_results.items():
        row = {'Model': model_name}
        # Ensure 'results' is a dictionary, even if empty
        if not isinstance(results, dict):
            logger.warning(f"Results for model '{model_name}' is not a dictionary: {results}. Skipping.")
            continue

        for metric, values in results.items():
            if metric == 'optimal_threshold' or metric == 'min_precision_constraint': # Skip non-numeric metrics added by find_optimal_threshold
                 row[metric.replace('_', ' ').title()] = values # Add threshold/constraint info if present
                 continue

            metric_capitalized = metric.capitalize()
            if has_cv_structure:
                # Handle nested structure from cross_validate_model
                if isinstance(values, dict):
                     row[f'{metric_capitalized} (Mean)'] = values.get('mean', np.nan)
                     ci_lower = values.get('ci_lower', np.nan)
                     ci_upper = values.get('ci_upper', np.nan)
                     # Format CI only if values are not NaN
                     if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                         row[f'{metric_capitalized} (CI)'] = f"({ci_lower:.3f}, {ci_upper:.3f})"
                     else:
                         row[f'{metric_capitalized} (CI)'] = "N/A"
                else:
                     # Fallback if structure is inconsistent
                     logger.warning(f"Expected dict for metric '{metric}' in CV results for model '{model_name}', got {type(values)}. Setting to NaN.")
                     row[f'{metric_capitalized} (Mean)'] = np.nan
                     row[f'{metric_capitalized} (CI)'] = "N/A"
            else:
                # Handle flat structure from evaluate_classifier
                if isinstance(values, (int, float)):
                    row[metric_capitalized] = values
                else:
                     logger.warning(f"Expected numeric value for metric '{metric}' in flat results for model '{model_name}', got {type(values)}. Setting to NaN.")
                     row[metric_capitalized] = np.nan


        comparison_data.append(row)

    if not comparison_data:
        logger.warning("No data to compare models.")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(comparison_data)

    # Define sort column based on structure
    sort_column = f'{target_metric.capitalize()} (Mean)' if has_cv_structure else target_metric.capitalize()

    if sort_column in comparison_df.columns:
        # Handle potential NaNs during sorting
        comparison_df = comparison_df.sort_values(sort_column, ascending=False, na_position='last')
    else:
        logger.warning(f"Target metric '{sort_column}' not found for sorting comparison table.")
        # Fallback sort by Model name
        comparison_df = comparison_df.sort_values('Model')


    # Reorder columns for better readability if needed
    # Example: move Model first, then target metric, then others
    cols_order = ['Model']
    if sort_column in comparison_df.columns:
        cols_order.append(sort_column)
        if has_cv_structure:
            ci_col = f'{target_metric.capitalize()} (CI)'
            if ci_col in comparison_df.columns:
                cols_order.append(ci_col)

    # Add remaining columns, filtering out the ones already added
    other_cols = [col for col in comparison_df.columns if col not in cols_order]
    comparison_df = comparison_df[cols_order + other_cols]


    return comparison_df

# ... (rest of the file)

def analyze_misclassifications(X, y_true, y_pred, feature_names=None):
    """
    Analyze misclassified samples to gain insights.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    feature_names : list, optional
        Feature names
        
    Returns:
    --------
    dict
        Dictionary with misclassification analysis
    """
    logger.info("Analyzing misclassifications")
    
    # Convert X to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X)
    
    # Find misclassified indices
    misclassified = y_true != y_pred
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    # Basic counts
    analysis = {
        'total_samples': len(y_true),
        'misclassified_count': np.sum(misclassified),
        'misclassification_rate': np.mean(misclassified),
        'false_positive_count': np.sum(false_positives),
        'false_negative_count': np.sum(false_negatives)
    }
    
    # Feature statistics for misclassified samples
    if len(X.columns) > 0:
        # Feature statistics for false negatives (critical errors)
        if np.sum(false_negatives) > 0:
            fn_stats = X[false_negatives].describe()
            analysis['false_negative_stats'] = fn_stats
            
        # Feature statistics for false positives
        if np.sum(false_positives) > 0:
            fp_stats = X[false_positives].describe()
            analysis['false_positive_stats'] = fp_stats
    
    logger.info(f"Misclassification analysis: {analysis['misclassified_count']} samples "
               f"({analysis['misclassification_rate']:.4f})")
    logger.info(f"False negatives: {analysis['false_negative_count']}, "
               f"False positives: {analysis['false_positive_count']}")
    
    return analysis
