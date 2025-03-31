import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging

logger = logging.getLogger(__name__)

def _simple_imputation(data):
    """Simple imputation using mean, median, or mode."""
    # For numerical features: median imputation
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        if data[col].isnull().any():
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)
            logger.info(f"Imputed missing values in '{col}' with median ({median_value})")
    
    # For categorical features: mode imputation
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if data[col].isnull().any():
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
            logger.info(f"Imputed missing values in '{col}' with mode ({mode_value})")
    
    return data

def _knn_imputation(data, n_neighbors=5):
    """KNN imputation based on feature similarity."""
    # Separate categorical and numerical data
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    cat_data = data[cat_cols]
    
    # Convert categorical data to one-hot encoding for KNN imputation
    if not cat_data.empty:
        for col in cat_cols:
            if data[col].isnull().any():
                # Mode imputation for categorical data
                data[col] = data[col].fillna(data[col].mode()[0])
    
    # Apply KNN imputation for numerical features
    num_data = data.select_dtypes(include=np.number)
    if not num_data.empty and num_data.isnull().any().any():
        # Create a copy to avoid SettingWithCopyWarning
        num_cols = num_data.columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data.loc[:, num_cols] = imputer.fit_transform(num_data)
        logger.info(f"Applied KNN imputation (n_neighbors={n_neighbors}) for numerical features")
    
    return data

def _mice_imputation(data, max_iter=10):
    """Multiple Imputation by Chained Equations (MICE)."""
    # Separate categorical features for separate handling
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    # Mode imputation for categorical features
    for col in cat_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # MICE imputation for numerical features
    num_data = data.select_dtypes(include=np.number)
    if not num_data.empty and num_data.isnull().any().any():
        num_cols = num_data.columns
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        data.loc[:, num_cols] = imputer.fit_transform(num_data)
        logger.info(f"Applied MICE imputation (max_iter={max_iter}) for numerical features")
    
    return data

def _indicator_imputation(data):
    """Add missing indicators and use median imputation."""
    # For numerical features: create missing indicators and apply median imputation
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        if data[col].isnull().any():
            # Create missing indicator
            data[f"{col}_missing"] = data[col].isnull().astype(int)
            logger.info(f"Created missing indicator for '{col}'")
            
            # Apply median imputation
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)
            logger.info(f"Imputed missing values in '{col}' with median ({median_value})")
    
    # For categorical features: create missing indicators and apply mode imputation
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if data[col].isnull().any():
            # Create missing indicator
            data[f"{col}_missing"] = data[col].isnull().astype(int)
            logger.info(f"Created missing indicator for '{col}'")
            
            # Apply mode imputation
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
            logger.info(f"Imputed missing values in '{col}' with mode ({mode_value})")
    
    return data

def _advanced_imputation(data, missingness_threshold=0.2):
    """
    Apply different imputation strategies based on data characteristics:
    - For features with low missingness: KNN imputation
    - For features with high missingness: Add missing indicators 
    - For categorical features: Mode imputation with missing indicators
    """
    # Calculate missingness per column
    missingness = data.isnull().mean()
    
    # Separate numerical and categorical columns
    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    
    # Add missing indicators for all features with missing values
    for col in data.columns:
        if data[col].isnull().any():
            data[f"{col}_missing"] = data[col].isnull().astype(int)
            logger.info(f"Created missing indicator for '{col}'")
    
    # Handle categorical features with mode imputation
    for col in cat_cols:
        if data[col].isnull().any():
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
            logger.info(f"Imputed missing values in '{col}' with mode ({mode_value})")
    
    # Handle numerical features based on missingness
    low_miss_num_cols = [col for col in num_cols if 0 < missingness[col] < missingness_threshold]
    high_miss_num_cols = [col for col in num_cols if missingness[col] >= missingness_threshold]
    
    # KNN imputation for numerical features with low missingness
    if low_miss_num_cols:
        imputer = KNNImputer(n_neighbors=5)
        data.loc[:, low_miss_num_cols] = imputer.fit_transform(data[low_miss_num_cols])
        logger.info(f"Applied KNN imputation for {len(low_miss_num_cols)} numerical features with low missingness")
    
    # Median imputation for numerical features with high missingness
    for col in high_miss_num_cols:
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)
        logger.info(f"Imputed missing values in '{col}' with median ({median_value}) due to high missingness")
    
    return data
