"""
Ensemble models for heart disease prediction.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import logging

logger = logging.getLogger(__name__)

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom weighted ensemble that combines multiple classifiers,
    prioritizing models with higher recall.
    """
    
    def __init__(self, classifiers=None, weights=None):
        """
        Initialize the ensemble.
        
        Parameters:
        -----------
        classifiers : list
            List of (name, classifier) tuples
        weights : list
            List of weights for each classifier
        """
        self.classifiers = classifiers if classifiers else []
        self.weights = weights if weights else [1] * len(self.classifiers)
        self.fitted_classifiers = None
        
    def fit(self, X, y):
        """
        Fit each classifier in the ensemble.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target values
            
        Returns:
        --------
        self
        """
        logger.info(f"Fitting weighted ensemble with {len(self.classifiers)} classifiers")
        
        self.fitted_classifiers = []
        self.classes_ = np.unique(y)
        
        for name, clf in self.classifiers:
            logger.info(f"Fitting {name} classifier")
            clf.fit(X, y)
            self.fitted_classifiers.append((name, clf))
            
        return self
        
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted average.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
            
        Returns:
        --------
        array
            Class probabilities
        """
        if not self.fitted_classifiers:
            raise ValueError("Estimator not fitted, call 'fit' first")
            
        # Collect predictions from each classifier
        probas = []
        for i, (name, clf) in enumerate(self.fitted_classifiers):
            logger.debug(f"Getting predictions from {name}")
            proba = clf.predict_proba(X)
            probas.append(proba * self.weights[i])
            
        # Weighted average
        avg_proba = np.sum(probas, axis=0) / np.sum(self.weights)
        
        return avg_proba
        
    def predict(self, X):
        """
        Predict class labels using weighted majority voting.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
            
        Returns:
        --------
        array
            Predicted class labels
        """
        avg_proba = self.predict_proba(X)
        return np.argmax(avg_proba, axis=1)

def create_voting_ensemble(estimators, voting='soft', weights=None):
    """
    Create a voting ensemble classifier.
    
    Parameters:
    -----------
    estimators : list
        List of (name, estimator) tuples
    voting : str, default='soft'
        Voting strategy: 'hard' or 'soft'
    weights : list, default=None
        Weights for each classifier
        
    Returns:
    --------
    VotingClassifier
        Voting ensemble classifier
    """
    logger.info(f"Creating voting ensemble with {len(estimators)} estimators and {voting} voting")
    
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights
    )

def create_stacking_ensemble(estimators, final_estimator=None):
    """
    Create a stacking ensemble classifier.
    
    Parameters:
    -----------
    estimators : list
        List of (name, estimator) tuples
    final_estimator : estimator, default=None
        Final estimator to combine predictions
        
    Returns:
    --------
    StackingClassifier
        Stacking ensemble classifier
    """
    logger.info(f"Creating stacking ensemble with {len(estimators)} base estimators")
    
    if final_estimator is None:
        final_estimator = LogisticRegression(class_weight='balanced')
        
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba'
    )

def create_recall_optimized_ensemble(estimators, X_train, y_train):
    """
    Create an ensemble optimized for recall.
    
    Parameters:
    -----------
    estimators : list
        List of (name, estimator) tuples
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    WeightedEnsembleClassifier
        Weighted ensemble classifier
    """
    logger.info(f"Creating recall-optimized ensemble with {len(estimators)} estimators")
    
    # Evaluate recall for each estimator
    recalls = []
    for name, estimator in estimators:
        logger.info(f"Evaluating {name} with cross-validation")
        try:
            # Use cross-validation to get predictions
            y_pred = cross_val_predict(estimator, X_train, y_train, cv=5)
            
            # Calculate recall
            recall = np.sum((y_train == 1) & (y_pred == 1)) / np.sum(y_train == 1)
            recalls.append(recall)
            
            logger.info(f"{name} cross-validated recall: {recall:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            recalls.append(0.0)
    
    # Convert recalls to weights (higher recall = higher weight)
    weights = np.array(recalls)
    
    # Add a minimum weight and normalize
    weights = weights + 0.1
    weights = weights / np.sum(weights)
    
    logger.info(f"Calculated weights based on recall: {list(zip([name for name, _ in estimators], weights))}")
    
    # Create the weighted ensemble
    ensemble = WeightedEnsembleClassifier(estimators, weights)
    
    # Fit all models before returning
    for name, estimator in estimators:
        logger.info(f"Fitting {name} for ensemble")
        estimator.fit(X_train, y_train)
    
    return ensemble

def calibrate_model_threshold(model, X_val, y_val, optimize_for='recall', min_precision=0.1):
    """
    Calibrate the probability threshold for classification.
    
    Parameters:
    -----------
    model : estimator
        Fitted model with predict_proba method
    X_val : array-like
        Validation features
    y_val : array-like
        Validation target
    optimize_for : str, default='recall'
        Metric to optimize ('recall', 'f1', or 'balanced')
    min_precision : float, default=0.1
        Minimum precision constraint
        
    Returns:
    --------
    float
        Optimal threshold
    """
    logger.info(f"Calibrating threshold to optimize {optimize_for} with min precision {min_precision}")
    
    # Get probability predictions
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Try different thresholds
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        true_positives = np.sum((y_val == 1) & (y_pred == 1))
        false_positives = np.sum((y_val == 0) & (y_pred == 1))
        false_negatives = np.sum((y_val == 1) & (y_pred == 0))
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Skip if precision is below minimum threshold
        if precision < min_precision:
            continue
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate score based on optimization goal
        if optimize_for == 'recall':
            score = recall
        elif optimize_for == 'f1':
            score = f1
        elif optimize_for == 'balanced':
            score = (recall + precision) / 2
        else:
            raise ValueError(f"Unknown optimization metric: {optimize_for}")
        
        # Update if this is the best score
        if score > best_score:
            best_score = score
            best_threshold = threshold
            logger.info(f"New best threshold: {threshold:.2f} with {optimize_for}={score:.4f} (precision={precision:.4f}, recall={recall:.4f})")
    
    logger.info(f"Final optimal threshold: {best_threshold:.4f} with {optimize_for}={best_score:.4f}")
    
    return best_threshold