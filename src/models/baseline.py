"""
Baseline majority classifier implementation.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that always predicts the majority class from the training data.
    This follows scikit-learn's estimator API.
    """
    
    def __init__(self):
        self.majority_class = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fit the classifier by determining the majority class.
        
        Parameters:
        -----------
        X : array-like
            Training data (ignored)
        y : array-like
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        logger.info("Fitting majority classifier")
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        # Count class occurrences and find majority
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
        
        majority_percentage = np.max(counts) / len(y) * 100
        logger.info(f"Majority class is {self.majority_class} "
                   f"({majority_percentage:.2f}% of the data)")
        
        return self
        
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : array-like
            Samples
            
        Returns:
        --------
        array
            Predicted classes
        """
        if self.majority_class is None:
            raise ValueError("Classifier has not been fitted. Call 'fit' first.")
            
        logger.info(f"Predicting majority class ({self.majority_class}) for {X.shape[0]} samples")
        return np.full(X.shape[0], self.majority_class)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like
            Samples
            
        Returns:
        --------
        array
            Predicted class probabilities
        """
        if self.majority_class is None:
            raise ValueError("Classifier has not been fitted. Call 'fit' first.")
            
        # Create probabilities array (all zeros)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        # Set probability 1.0 for majority class
        majority_idx = np.where(self.classes_ == self.majority_class)[0][0]
        proba[:, majority_idx] = 1.0
        
        return proba