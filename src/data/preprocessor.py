"""
Data preprocessing functions for the heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

from .imputation import _simple_imputation, _knn_imputation, _mice_imputation, _indicator_imputation, _advanced_imputation

logger = logging.getLogger(__name__)

def check_data_quality(data):
    """
    Check data quality and return summary statistics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
        
    Returns:
    --------
    dict
        Dictionary containing data quality summary
    """
    logger.info("Checking data quality")
    
    # Create summary
    summary = {
        "shape": data.shape,
        "missing_values": data.isnull().sum().to_dict(),
        "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
        "duplicates": data.duplicated().sum(),
        "data_types": data.dtypes.to_dict()
    }
    
    logger.info(f"Data shape: {summary['shape']}")
    logger.info(f"Number of duplicates: {summary['duplicates']}")
    
    # Log columns with missing values
    cols_with_missing = {k: v for k, v in summary["missing_values"].items() if v > 0}
    if cols_with_missing:
        logger.info(f"Columns with missing values: {cols_with_missing}")
    else:
        logger.info("No missing values found")
        
    return summary

def clean_data(data, method='advanced'):
    """
    Clean the dataset by handling duplicates, converting data types,
    and applying sophisticated missing data handling techniques.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    method : str, default='advanced'
        Method for handling missing values:
        - 'simple': Mean/median/mode imputation
        - 'knn': K-Nearest Neighbors imputation
        - 'mice': Multiple Imputation by Chained Equations
        - 'indicator': Add missing indicator variables 
        - 'advanced': Combination of methods based on data characteristics
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset with handled missing values
    """
    logger.info("Cleaning data")
    
    # Make a copy of the data
    cleaned_data = data.copy()
    
    # Remove duplicates if any
    if cleaned_data.duplicated().any():
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(cleaned_data)} duplicate rows")
    
    # Convert 'Yes'/'No' values to 1/0 for the target variable
    if "Heart Disease Status" in cleaned_data.columns:
        cleaned_data["Heart Disease Status"] = cleaned_data["Heart Disease Status"].map({"Yes": 1, "No": 0})
        logger.info("Converted target variable to binary (1/0)")
    
    # Handle missing values based on the chosen method
    if method == 'simple':
        cleaned_data = _simple_imputation(cleaned_data)
    elif method == 'knn':
        cleaned_data = _knn_imputation(cleaned_data)
    elif method == 'mice':
        cleaned_data = _mice_imputation(cleaned_data)
    elif method == 'indicator':
        cleaned_data = _indicator_imputation(cleaned_data)
    elif method == 'advanced':
        cleaned_data = _advanced_imputation(cleaned_data)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    return cleaned_data

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Parameters:
    -----------
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline
    """
    logger.info("Creating preprocessing pipeline")
    
    # Numeric features preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    logger.info(f"Preprocessing pipeline created for {len(numerical_features)} numerical and "
                f"{len(categorical_features)} categorical features")
    
    return preprocessor

def encode_target(y):
    """
    Encode the target variable to numeric if needed.
    
    Parameters:
    -----------
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    pandas.Series
        Encoded target variable
    """
    if y.dtype == 'object':
        logger.info("Encoding target variable")
        # Map Yes/No to 1/0
        return y.map({"Yes": 1, "No": 0})
    return y