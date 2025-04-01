"""
Hyperparameter tuning functions for heart disease prediction models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)

def create_recall_scorer():
    """
    Create a custom scorer that prioritizes recall.
    
    Returns:
    --------
    dict
        Dictionary of scorers
    """
    return {
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }

def tune_logistic_regression(X_train, y_train, cv=5, class_weight='balanced'):
    """
    Tune hyperparameters for logistic regression.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    class_weight : str or dict, default='balanced'
        Class weights
        
    Returns:
    --------
    LogisticRegression
        Best logistic regression model
    """
    logger.info("Tuning logistic regression hyperparameters")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['saga'],  # saga supports all penalties
        'l1_ratio': [0.1, 0.5, 0.9],  # only used with elasticnet
        'max_iter': [1000],
        'class_weight': [class_weight]
    }
    
    # Create base model
    lr = LogisticRegression(random_state=42)
    
    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the grid search
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=create_recall_scorer(),
        refit='recall',  # Optimize for recall
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Best logistic regression parameters: {best_params}")
    logger.info(f"Best recall score: {best_score:.4f}")
    
    return best_model

def tune_random_forest(X_train, y_train, cv=5, class_weight='balanced'):
    """
    Tune hyperparameters for random forest.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    class_weight : str or dict, default='balanced'
        Class weights
        
    Returns:
    --------
    RandomForestClassifier
        Best random forest model
    """
    logger.info("Tuning random forest hyperparameters")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': [class_weight],
        'criterion': ['gini', 'entropy']
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the grid search
    grid_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        cv=cv_strategy,
        scoring=create_recall_scorer(),
        refit='recall',  # Optimize for recall
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Best random forest parameters: {best_params}")
    logger.info(f"Best recall score: {best_score:.4f}")
    
    return best_model

def tune_xgboost(X_train, y_train, cv=5):
    """
    Tune hyperparameters for XGBoost.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    XGBClassifier
        Best XGBoost model
    """
    logger.info("Tuning XGBoost hyperparameters")
    
    # Calculate scale_pos_weight for imbalanced dataset
    if isinstance(y_train, pd.Series):
        y_train_values = y_train.values
    else:
        y_train_values = y_train
        
    negative_samples = np.sum(y_train_values == 0)
    positive_samples = np.sum(y_train_values == 1)
    scale_pos_weight = negative_samples / positive_samples if positive_samples > 0 else 1.0
    
    logger.info(f"Using scale_pos_weight={scale_pos_weight} for imbalanced data")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    # Create base model
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    )
    
    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the grid search
    grid_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        cv=cv_strategy,
        scoring=create_recall_scorer(),
        refit='recall',  # Optimize for recall
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Best XGBoost parameters: {best_params}")
    logger.info(f"Best recall score: {best_score:.4f}")
    
    return best_model

def tune_gradient_boosting(X_train, y_train, cv=5):
    """
    Tune hyperparameters for Gradient Boosting.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    GradientBoostingClassifier
        Best Gradient Boosting model
    """
    logger.info("Tuning Gradient Boosting hyperparameters")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create the grid search
    grid_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        cv=cv_strategy,
        scoring=create_recall_scorer(),
        refit='recall',  # Optimize for recall
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Best Gradient Boosting parameters: {best_params}")
    logger.info(f"Best recall score: {best_score:.4f}")
    
    return best_model