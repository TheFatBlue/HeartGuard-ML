"""
Functions to load the heart disease dataset.
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load the heart disease data from CSV file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    try:
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_features_target(data, target_column):
    """
    Split the dataset into features and target.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    target_column : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    try:
        logger.info(f"Splitting data into features and target '{target_column}'")
        X = data.drop(columns=[target_column])
        y = data[target_column]
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def save_processed_data(data, filepath):
    """
    Save processed data to a CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed data to save
    filepath : str or Path
        Path where to save the CSV file
    """
    try:
        logger.info(f"Saving processed data to {filepath}")
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Successfully saved processed data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise