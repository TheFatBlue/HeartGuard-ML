"""
Feature engineering functions for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class HealthRiskScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Create a health risk score based on multiple risk factors.
    This class follows scikit-learn's transformer interface.
    """
    
    def __init__(self):
        self.risk_factors = {
            'Age': {'weight': 0.05, 'threshold': 50},
            'Blood Pressure': {'weight': 0.1, 'threshold': 140},
            'Cholesterol Level': {'weight': 0.1, 'threshold': 200},
            'BMI': {'weight': 0.1, 'threshold': 25},
            'Smoking': {'weight': 0.15, 'value': 'Yes'},
            'Family Heart Disease': {'weight': 0.15, 'value': 'Yes'},
            'Diabetes': {'weight': 0.15, 'value': 'Yes'},
            'High Blood Pressure': {'weight': 0.1, 'value': 'Yes'},
            'Sleep Hours': {'weight': 0.05, 'threshold': 7, 'inverse': True},
            'Stress Level': {'weight': 0.05, 'value': 'High'}
        }
        
    def fit(self, X, y=None):
        """
        No fitting required for this transformer.
        """
        return self
        
    def transform(self, X):
        """
        Transform the data by adding a health risk score.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data with additional health risk score
        """
        logger.info("Creating health risk score feature")
        X_copy = X.copy()
        
        # Initialize risk score
        X_copy['health_risk_score'] = 0
        
        # Calculate risk score based on risk factors
        for factor, params in self.risk_factors.items():
            if factor in X_copy.columns:
                if 'threshold' in params:
                    # For numerical features
                    if params.get('inverse', False):
                        # Lower values are worse (e.g., sleep hours)
                        contribution = np.where(X_copy[factor] < params['threshold'], 
                                               params['weight'], 0)
                    else:
                        # Higher values are worse
                        contribution = np.where(X_copy[factor] > params['threshold'], 
                                               params['weight'], 0)
                elif 'value' in params:
                    # For categorical features
                    contribution = np.where(X_copy[factor] == params['value'], 
                                           params['weight'], 0)
                
                X_copy['health_risk_score'] += contribution
        
        logger.info("Health risk score feature created")
        return X_copy

def create_interaction_features(data):
    """
    Create interaction features that might be relevant for heart disease prediction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
        
    Returns:
    --------
    pandas.DataFrame
        Data with additional interaction features
    """
    logger.info("Creating interaction features")
    
    # Make a copy of the data
    enhanced_data = data.copy()
    
    # Age and Blood Pressure interaction
    if 'Age' in data.columns and 'Blood Pressure' in data.columns:
        enhanced_data['age_bp_interaction'] = data['Age'] * data['Blood Pressure'] / 100
        logger.info("Created Age x Blood Pressure interaction feature")
    
    # BMI and Cholesterol interaction
    if 'BMI' in data.columns and 'Cholesterol Level' in data.columns:
        enhanced_data['bmi_cholesterol_interaction'] = data['BMI'] * data['Cholesterol Level'] / 100
        logger.info("Created BMI x Cholesterol interaction feature")
    
    # Age and Diabetes interaction
    if 'Age' in data.columns and 'Diabetes' in data.columns:
        # Convert Diabetes to binary if it's not already
        diabetes_binary = data['Diabetes'].map({'Yes': 1, 'No': 0}) if data['Diabetes'].dtype == 'object' else data['Diabetes']
        enhanced_data['age_diabetes_interaction'] = data['Age'] * diabetes_binary
        logger.info("Created Age x Diabetes interaction feature")
    
    # Smoking and Cholesterol interaction
    if 'Smoking' in data.columns and 'Cholesterol Level' in data.columns:
        # Convert Smoking to binary if it's not already
        smoking_binary = data['Smoking'].map({'Yes': 1, 'No': 0}) if data['Smoking'].dtype == 'object' else data['Smoking']
        enhanced_data['smoking_cholesterol_interaction'] = smoking_binary * data['Cholesterol Level']
        logger.info("Created Smoking x Cholesterol interaction feature")
    
    logger.info(f"Created {len(enhanced_data.columns) - len(data.columns)} new interaction features")
    
    return enhanced_data

def compute_lipid_ratio(data):
    """
    Compute the lipid ratio (Total Cholesterol / HDL) if possible.
    This is a known risk factor for heart disease.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
        
    Returns:
    --------
    pandas.DataFrame
        Data with additional lipid ratio feature
    """
    logger.info("Computing lipid ratio feature")
    
    # Make a copy of the data
    result = data.copy()
    
    # We don't have direct HDL values, but we have a indicator for low HDL
    # We can create a proxy feature based on cholesterol and LDL status
    if ('Cholesterol Level' in data.columns and 
        'Low HDL Cholesterol' in data.columns and 
        'High LDL Cholesterol' in data.columns):
        
        # Convert categorical variables to binary if needed
        low_hdl = data['Low HDL Cholesterol'].map({'Yes': 1, 'No': 0}) if data['Low HDL Cholesterol'].dtype == 'object' else data['Low HDL Cholesterol']
        high_ldl = data['High LDL Cholesterol'].map({'Yes': 1, 'No': 0}) if data['High LDL Cholesterol'].dtype == 'object' else data['High LDL Cholesterol']
        
        # Create a lipid risk factor
        result['lipid_risk_factor'] = data['Cholesterol Level'] * (1 + low_hdl) * (1 + high_ldl) / 100
        logger.info("Created lipid_risk_factor feature")
    
    return result

def get_feature_names(column_transformer):
    """
    Get feature names from a column transformer.
    
    Parameters:
    -----------
    column_transformer : ColumnTransformer
        Fitted column transformer
        
    Returns:
    --------
    list
        List of feature names
    """
    try:
        # First try the get_feature_names_out method (scikit-learn >= 1.0)
        if hasattr(column_transformer, 'get_feature_names_out'):
            return column_transformer.get_feature_names_out()
    except Exception as e:
        logger.warning(f"Could not use get_feature_names_out: {e}")
        
    # Fallback to manual extraction
    feature_names = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(pipe, 'get_feature_names_out'):
                # For newer scikit-learn versions
                feature_names.extend(pipe.get_feature_names_out(features))
            elif hasattr(pipe, 'steps') and hasattr(pipe.steps[-1][1], 'get_feature_names_out'):
                # For transformers within a pipeline
                feature_names.extend(pipe.steps[-1][1].get_feature_names_out(features))
            elif hasattr(pipe, 'categories_'):
                # For OneHotEncoder
                for i, category in enumerate(pipe.categories_):
                    feature_names.extend([f"{features[i]}_{cat}" for cat in category])
            else:
                # Fallback for transformers without get_feature_names_out
                feature_names.extend(features)
    
    if not feature_names:
        logger.warning("Could not extract feature names, using generic names")
        feature_names = [f"feature_{i}" for i in range(column_transformer.transform(pd.DataFrame()).shape[1])]
                
    return feature_names