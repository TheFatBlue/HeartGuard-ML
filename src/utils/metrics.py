"""
Custom evaluation metrics for heart disease prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)

def evaluate_classifier(y_true, y_pred, y_proba=None):
    """
    Evaluate a classifier with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    y_proba : array-like, optional
        Predicted probabilities for the positive class
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating classifier performance")
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    # If probabilities are provided, calculate ROC AUC
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    logger.info(f"Classification metrics: accuracy={metrics['accuracy']:.4f}, "
               f"recall={metrics['recall']:.4f}, precision={metrics['precision']:.4f}, "
               f"f1={metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def find_optimal_threshold(y_true, y_proba, optimize_for='recall', min_precision=0.5):
    """
    Find the optimal probability threshold for classification.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_proba : array-like
        Predicted probabilities for the positive class
    optimize_for : str, default='recall'
        Metric to optimize ('recall', 'f1', or 'balanced')
    min_precision : float, default=0.5
        Minimum precision constraint
        
    Returns:
    --------
    float
        Optimal threshold
    """
    logger.info(f"Finding optimal threshold to optimize {optimize_for} "
               f"with minimum precision {min_precision}")
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Add threshold=1.0 for the last point in precisions/recalls
    thresholds = np.append(thresholds, 1.0)
    
    # Calculate F1 scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Filter by minimum precision constraint
    valid_indices = precisions >= min_precision
    
    if not np.any(valid_indices):
        logger.warning(f"No threshold satisfies the minimum precision of {min_precision}")
        min_precision = precisions.max() * 0.9
        valid_indices = precisions >= min_precision
        logger.warning(f"Relaxed minimum precision to {min_precision:.4f}")
    
    valid_recalls = recalls[valid_indices]
    valid_f1 = f1_scores[valid_indices]
    valid_thresholds = thresholds[valid_indices]
    valid_precisions = precisions[valid_indices]
    
    # Find optimal threshold based on optimization goal
    if optimize_for == 'recall':
        best_idx = np.argmax(valid_recalls)
    elif optimize_for == 'f1':
        best_idx = np.argmax(valid_f1)
    elif optimize_for == 'balanced':
        # Balanced option: optimize (recall + precision) / 2
        balanced_score = (valid_recalls + valid_precisions) / 2
        best_idx = np.argmax(balanced_score)
    else:
        raise ValueError(f"Unknown optimization metric: {optimize_for}")
    
    optimal_threshold = valid_thresholds[best_idx]
    optimal_recall = valid_recalls[best_idx]
    optimal_precision = valid_precisions[best_idx]
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} "
               f"(recall={optimal_recall:.4f}, precision={optimal_precision:.4f})")
    
    return optimal_threshold

def compare_models(model_results, target_metric='recall'):
    """
    Compare multiple models based on evaluation metrics.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and evaluation results as values
    target_metric : str, default='recall'
        Primary metric for comparison
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table of all models
    """
    logger.info(f"Comparing models based on {target_metric}")
    
    # Extract metrics for each model
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Recall': results['recall'],
            'Precision': results['precision'],
            'F1': results['f1'],
            'False Negatives': results['false_negatives']
        }
        
        if 'roc_auc' in results:
            row['ROC AUC'] = results['roc_auc']
            
        comparison_data.append(row)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by target metric
    comparison_df = comparison_df.sort_values(target_metric.capitalize(), ascending=False)
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    logger.info(f"Best model based on {target_metric}: {best_model} "
               f"({target_metric}={comparison_df.iloc[0][target_metric.capitalize()]:.4f})")
    
    return comparison_df

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