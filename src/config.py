"""
Configuration parameters for the heart disease prediction project.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart_disease_data.csv"
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

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

# Class imbalance handling
SMOTE_PARAMS = {
    "random_state": RANDOM_STATE,
    "k_neighbors": 5
}

# Evaluation metrics
SCORING = ["recall", "precision", "f1", "roc_auc"]