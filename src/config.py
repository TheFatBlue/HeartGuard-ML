"""
Configuration parameters for the heart disease prediction project.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_disease.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_disease_cleaned.csv"

# Data preprocessing
TARGET_COLUMN = "Heart Disease Status"
CATEGORICAL_FEATURES = [
    "Gender", 
    "Exercise Habits", 
    "Smoking", 
    "Family Heart Disease",
    "Diabetes", 
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol", 
    "Alcohol Consumption",
    "Stress Level",
    "Sugar Consumption"
]
NUMERICAL_FEATURES = [
    "Age", 
    "Blood Pressure", 
    "Cholesterol Level", 
    "BMI",
    "Sleep Hours", 
    "Triglyceride Level", 
    "Fasting Blood Sugar",
    "CRP Level", 
    "Homocysteine Level"
]

# Ensure all features are accounted for
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Random state for reproducibility
RANDOM_STATE = 42

# Cross-validation parameters
CV_FOLDS = 5

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "max_iter": 1000
}

LOGISTIC_REGRESSION_GRID = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0], # Explore a range of regularization strengths
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear', 'saga'], # Solvers compatible with l1/l2
    'classifier__class_weight': ['balanced', {0:1, 1:5}, {0:1, 1:10}] # Test different weighting schemes
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300, # Set a fixed value if removed from grid
    "max_depth": None, # Default, will be overridden by GridSearch if tuned
    "min_samples_split": 2,
    "min_samples_leaf": 2, # Default, will be overridden by GridSearch if tuned
    "bootstrap": True,
    "class_weight": "balanced", # Default, will be overridden by GridSearch if tuned
    "random_state": RANDOM_STATE
}

# Class imbalance handling
SMOTE_PARAMS = {
    "random_state": RANDOM_STATE,
    "k_neighbors": 5
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'learning_rate': 0.1, # Default, will be overridden by GridSearch if tuned
    'max_depth': 5,       # Default, will be overridden by GridSearch if tuned
    'min_child_weight': 1, # Set fixed value if removed from grid
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1, # Default, will be overridden by GridSearch if tuned
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_GRID = { # [cite: 2]
    # 'classifier__n_estimators': [200, 500],
    'classifier__max_depth': [10, 50, None],     # Tune max depth
    'classifier__min_samples_leaf': [2, 4],    # Tune min samples leaf
    # 'classifier__class_weight': ['balanced', 'balanced_subsample', {0:1, 1:5}, {0:1, 1:10}] # Tune class weight
}

XGBOOST_GRID = { # [cite: 5]
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.3],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__scale_pos_weight': [1, 4, 5] # Adjust based on class imbalance
    # Add gamma, subsample, colsample_bytree etc. if needed
}


# Ensemble configuration
VOTING_WEIGHTS = {'Logistic Regression': 0.2, 'Random Forest': 0.4, 'XGBoost': 0.4} # Example weights, potentially based on recall [cite: 8]

# Class imbalance handling (keep existing)
# ...

# Evaluation metrics
# Focus on recall, add PR AUC, F-beta
SCORING = ["recall", "precision", "f1", "roc_auc", "average_precision"] # average_precision is PR AUC
F_BETA_SCORE_BETA = 2 # Beta > 1 favors recall [cite: 16]

# Threshold optimization parameters [cite: 16]
THRESHOLD_MIN_PRECISION_CONSTRAINTS = [0.21, 0.25, 0.3, 0.35, 0.4, 0.45, 0.6]
DEFAULT_MIN_PRECISION = 0.21 # Default constraint if grid search is not used

# Fairness analysis parameters
FAIRNESS_SLICES = ['Age_Band', 'Gender', 'Diabetes'] # Example demographic slices [cite: 18]
AGE_BANDS = [(0, 40), (40, 60), (60, 100)] # Example age bands [cite: 9]
