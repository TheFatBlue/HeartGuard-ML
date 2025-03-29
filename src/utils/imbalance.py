"""
Utilities for handling class imbalance in the heart disease dataset.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)

def apply_smote(X, y, **kwargs):
    """
    Apply SMOTE to oversample the minority class.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    **kwargs : dict
        Additional parameters for SMOTE
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled)
    """
    logger.info("Applying SMOTE to handle class imbalance")
    
    try:
        # Ensure X and y have compatible shapes
        logger.info(f"SMOTE input shapes - X: {X.shape}, y: {y.shape}")
        if isinstance(y, pd.Series):
            y = y.values
        
        # Initial class distribution
        unique, counts = np.unique(y, return_counts=True)
        initial_distribution = dict(zip(unique, counts))
        logger.info(f"Initial class distribution: {initial_distribution}")
        
        # Apply SMOTE
        smote = SMOTE(**kwargs)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # New class distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        new_distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution after SMOTE: {new_distribution}")
        logger.info(f"SMOTE output shapes - X_resampled: {X_resampled.shape}, y_resampled: {y_resampled.shape}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"Error applying SMOTE: {e}")
        logger.error(f"X type: {type(X)}, y type: {type(y)}")
        if hasattr(X, 'shape'):
            logger.error(f"X shape: {X.shape}")
        if hasattr(y, 'shape'):
            logger.error(f"y shape: {y.shape}")
        # Fall back to original data
        logger.info("Falling back to original data without SMOTE")
        return X, y

def apply_undersampling(X, y, sampling_strategy=0.5, random_state=42):
    """
    Apply random undersampling to the majority class.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    sampling_strategy : float or dict, default=0.5
        Sampling strategy for RandomUnderSampler
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled)
    """
    logger.info(f"Applying undersampling with sampling_strategy={sampling_strategy}")
    
    # Initial class distribution
    unique, counts = np.unique(y, return_counts=True)
    initial_distribution = dict(zip(unique, counts))
    logger.info(f"Initial class distribution: {initial_distribution}")
    
    # Apply undersampling
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    
    # New class distribution
    unique, counts = np.unique(y_resampled, return_counts=True)
    new_distribution = dict(zip(unique, counts))
    logger.info(f"Class distribution after undersampling: {new_distribution}")
    
    return X_resampled, y_resampled

def get_class_weights(y):
    """
    Compute class weights to handle class imbalance.
    
    Parameters:
    -----------
    y : array-like
        Target values
        
    Returns:
    --------
    dict
        Dictionary with class weights
    """
    logger.info("Computing class weights")
    
    # Compute class weights
    unique_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y
    )
    
    # Create dictionary
    class_weights = dict(zip(unique_classes, weights))
    
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights

def create_cost_sensitive_sample_weights(y, false_negative_cost=10, false_positive_cost=1):
    """
    Create sample weights based on misclassification costs.
    Higher weights for minority class to minimize false negatives.
    
    Parameters:
    -----------
    y : array-like
        Target values
    false_negative_cost : float, default=10
        Cost of false negatives
    false_positive_cost : float, default=1
        Cost of false positives
        
    Returns:
    --------
    array
        Sample weights
    """
    logger.info(f"Creating cost-sensitive sample weights "
               f"(FN cost={false_negative_cost}, FP cost={false_positive_cost})")
    
    # Initialize weights
    sample_weights = np.ones(len(y))
    
    # Assign higher weights to positive samples (minority class)
    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]
    
    sample_weights[positive_idx] = false_negative_cost
    sample_weights[negative_idx] = false_positive_cost
    
    # Normalize weights to have mean=1
    sample_weights = sample_weights * len(y) / np.sum(sample_weights)
    
    logger.info(f"Created sample weights: min={sample_weights.min():.2f}, "
               f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
    
    return sample_weights