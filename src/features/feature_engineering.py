"""
Feature engineering functions for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
import logging

from src.config import AGE_BANDS

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

def create_age_bands(data):
    """Creates age bands based on config."""
    logger.info("Creating age bands")
    if 'Age' in data.columns:
        labels = [f"{low}-{high}" for low, high in AGE_BANDS]
        bins = [b[0] for b in AGE_BANDS] + [AGE_BANDS[-1][1]]
        data['Age_Band'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
        logger.info(f"Created Age_Band feature with bands: {labels}")
    else:
        logger.warning("Age column not found, cannot create age bands.")
    return data

def calculate_framingham_score(data):
    """Placeholder function to calculate Framingham Risk Score."""
    logger.info("Calculating Framingham Risk Score (Placeholder)")
    # This requires specific coefficients based on gender, age, cholesterol, HDL, BP, smoking, diabetes.
    # Implementation depends on the exact FRS version used.
    # Returning a dummy score for now.
    if all(col in data.columns for col in ['Age', 'Cholesterol Level', 'Blood Pressure', 'Smoking']):
        # Example: Simplified score (NOT CLINICALLY VALID)
        data['Framingham_Score_Est'] = (data['Age']/10 +
                                       data['Cholesterol Level']/50 +
                                       data['Blood Pressure']/20 +
                                       data['Smoking'].map({'Yes': 1, 'No': 0}) * 2)
        logger.info("Created Framingham_Score_Est feature (placeholder).")
    else:
         logger.warning("Required columns for Framingham Score not found.")
    return data

def calculate_ascvd_score(data):
    """Placeholder function to calculate ASCVD Risk Score."""
    logger.info("Calculating ASCVD Risk Score (Placeholder)")
    # Requires specific coefficients based on race, gender, age, cholesterol, HDL, BP, diabetes, smoking.
    # Implementation depends on the specific ASCVD Pooled Cohort Equations.
    # Returning a dummy score for now.
    if all(col in data.columns for col in ['Age', 'Cholesterol Level', 'Blood Pressure', 'Diabetes', 'Smoking']):
         # Example: Simplified score (NOT CLINICALLY VALID)
        data['ASCVD_Score_Est'] = (data['Age']/10 +
                                   data['Cholesterol Level']/60 +
                                   data['Blood Pressure']/15 +
                                   data['Diabetes'].map({'Yes': 1, 'No': 0}) * 1.5 +
                                   data['Smoking'].map({'Yes': 1, 'No': 0}) * 1.5)
        logger.info("Created ASCVD_Score_Est feature (placeholder).")
    else:
        logger.warning("Required columns for ASCVD Score not found.")
    return data

def add_polynomial_features(data, numerical_features, degree=2):
    """Generate polynomial features for specified numerical columns."""
    logger.info(f"Generating polynomial features (degree={degree})")
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    
    # Select only specified numerical features that exist in the data
    valid_numerical_features = [f for f in numerical_features if f in data.columns]
    if not valid_numerical_features:
        logger.warning("No valid numerical features found for polynomial feature generation.")
        return data
        
    poly_features = poly.fit_transform(data[valid_numerical_features])
    poly_feature_names = poly.get_feature_names_out(valid_numerical_features)

    # Create a DataFrame with new features
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)

    # Drop original columns used to create polynomial features to avoid multicollinearity
    # (unless interaction_only=True was used, but here we allow powers too)
    # Add only the NEW features (higher order and interactions)
    new_feature_cols = [col for col in poly_feature_names if col not in valid_numerical_features]
    data = pd.concat([data, poly_df[new_feature_cols]], axis=1)

    logger.info(f"Added {len(new_feature_cols)} polynomial features.")
    return data

def map_feature_indices_to_names(engineered_features, original_features, transformer):
    """Maps engineered feature indices back to human-readable names."""
    # This is complex and highly dependent on the exact transformers used.
    # The `get_feature_names_out` method of scikit-learn pipelines/transformers
    # is the standard way. We already use this in `get_feature_names`.
    logger.info("Mapping feature indices to names (using get_feature_names_out)")
    # Assuming `transformer` is the fitted preprocessor or pipeline step
    try:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names = transformer.get_feature_names_out()
            return feature_names
        else:
            logger.warning("Transformer does not have 'get_feature_names_out'. Returning basic names.")
            # Fallback or more specific logic needed here based on transformer type
            return [f"feature_{i}" for i in range(engineered_features.shape[1])]
    except Exception as e:
        logger.error(f"Error getting feature names: {e}")
        return [f"feature_{i}" for i in range(engineered_features.shape[1])]

# Update get_feature_names function if necessary (current one seems reasonable)
def get_feature_names(column_transformer):
    """Get feature names from a column transformer."""
    # ... (Keep existing implementation, it tries get_feature_names_out first)
    # Ensure it handles polynomial features if they are added *before* the main preprocessor
    logger.info("Extracting feature names using get_feature_names method")
    try:
        # Use get_feature_names_out which is preferred in modern scikit-learn
        feature_names = list(column_transformer.get_feature_names_out())
        logger.info(f"Successfully extracted {len(feature_names)} feature names via get_feature_names_out.")
        return feature_names
    except AttributeError:
        logger.warning("'get_feature_names_out' not available, attempting manual extraction.")
        # Fallback logic (keep existing or adapt as needed)
        # ... (existing fallback logic) ...
    except Exception as e:
        logger.error(f"An unexpected error occurred during feature name extraction: {e}")
        # Fallback to generic names
        # Determine the number of features by transforming a dummy row
        try:
            num_features = column_transformer.transform(pd.DataFrame(columns=column_transformer.feature_names_in_)).shape[1]
        except:
            num_features = 20 # Arbitrary fallback
        logger.warning(f"Using {num_features} generic feature names as fallback.")
        return [f"feature_{i}" for i in range(num_features)]

def _get_feature_names(column_transformer):
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