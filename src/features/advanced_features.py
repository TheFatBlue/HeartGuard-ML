"""
Advanced feature engineering functions for heart disease prediction.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class MedicalRiskScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Create medical risk scores based on clinical guidelines.
    Follows scikit-learn's transformer interface.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'Age': {'high': 65, 'moderate': 45},
            'Blood Pressure': {'high': 140, 'moderate': 120},
            'Cholesterol Level': {'high': 240, 'moderate': 200},
            'BMI': {'high': 30, 'moderate': 25},
            'Triglyceride Level': {'high': 200, 'moderate': 150},
            'Fasting Blood Sugar': {'high': 126, 'moderate': 100}
        }
        
    def fit(self, X, y=None):
        """No fitting required."""
        return self
        
    def transform(self, X):
        """
        Transform the data by adding medical risk scores.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data with additional risk scores
        """
        logger.info("Creating medical risk score features")
        X_copy = X.copy()
        
        # Age risk score (higher age = higher risk)
        if 'Age' in X_copy.columns:
            X_copy['age_risk'] = np.where(
                X_copy['Age'] >= self.risk_thresholds['Age']['high'], 2,
                np.where(X_copy['Age'] >= self.risk_thresholds['Age']['moderate'], 1, 0)
            )
            
        # Blood pressure risk
        if 'Blood Pressure' in X_copy.columns:
            X_copy['bp_risk'] = np.where(
                X_copy['Blood Pressure'] >= self.risk_thresholds['Blood Pressure']['high'], 2,
                np.where(X_copy['Blood Pressure'] >= self.risk_thresholds['Blood Pressure']['moderate'], 1, 0)
            )
            
        # Cholesterol risk
        if 'Cholesterol Level' in X_copy.columns:
            X_copy['cholesterol_risk'] = np.where(
                X_copy['Cholesterol Level'] >= self.risk_thresholds['Cholesterol Level']['high'], 2,
                np.where(X_copy['Cholesterol Level'] >= self.risk_thresholds['Cholesterol Level']['moderate'], 1, 0)
            )
            
        # BMI risk
        if 'BMI' in X_copy.columns:
            X_copy['bmi_risk'] = np.where(
                X_copy['BMI'] >= self.risk_thresholds['BMI']['high'], 2,
                np.where(X_copy['BMI'] >= self.risk_thresholds['BMI']['moderate'], 1, 0)
            )
            
        # Diabetes-related risk (combining fasting glucose and diabetes status)
        if 'Fasting Blood Sugar' in X_copy.columns:
            X_copy['glucose_risk'] = np.where(
                X_copy['Fasting Blood Sugar'] >= self.risk_thresholds['Fasting Blood Sugar']['high'], 2,
                np.where(X_copy['Fasting Blood Sugar'] >= self.risk_thresholds['Fasting Blood Sugar']['moderate'], 1, 0)
            )
            
        # Combine diabetes risk factors
        if 'glucose_risk' in X_copy.columns and 'Diabetes' in X_copy.columns:
            # Convert categorical Diabetes to numeric if needed
            if X_copy['Diabetes'].dtype == 'object':
                diabetes_numeric = X_copy['Diabetes'].map({'Yes': 1, 'No': 0})
            else:
                diabetes_numeric = X_copy['Diabetes']
                
            # Combine glucose risk with diabetes status
            X_copy['diabetes_combined_risk'] = X_copy['glucose_risk'] + diabetes_numeric
            
        # Create combined cardiovascular risk score
        risk_columns = [col for col in ['age_risk', 'bp_risk', 'cholesterol_risk', 
                                        'bmi_risk', 'diabetes_combined_risk'] 
                        if col in X_copy.columns]
        
        if risk_columns:
            X_copy['cv_risk_score'] = X_copy[risk_columns].sum(axis=1)
            logger.info(f"Created cardiovascular risk score from {len(risk_columns)} risk factors")
            
        return X_copy

def create_polynomial_features(X, degree=2, interaction_only=True):
    """
    Create polynomial and interaction features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features
    degree : int, default=2
        Degree of polynomial features
    interaction_only : bool, default=True
        Whether to include only interaction terms
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with polynomial features
    """
    logger.info(f"Creating polynomial features (degree={degree}, interaction_only={interaction_only})")
    
    # Select only numerical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_features:
        logger.warning("No numerical features found for polynomial feature creation")
        return X
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(X[numeric_features])
    
    # Generate feature names
    if hasattr(poly, 'get_feature_names_out'):
        # For scikit-learn >= 1.0
        feature_names = poly.get_feature_names_out(numeric_features)
    else:
        # For older scikit-learn versions
        feature_names = poly.get_feature_names(numeric_features)
    
    # Create DataFrame with new features
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
    
    # Remove original features from polynomial DataFrame
    for feature in numeric_features:
        if feature in poly_df.columns:
            poly_df = poly_df.drop(columns=[feature])
    
    # Combine with original DataFrame
    result = pd.concat([X, poly_df], axis=1)
    
    logger.info(f"Created {poly_df.shape[1]} new polynomial features")
    
    return result

def create_risk_clusters(X):
    """
    Create features based on combinations of risk factors.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with risk cluster features
    """
    logger.info("Creating risk cluster features")
    
    result = X.copy()
    
    # Convert categorical features to binary if needed
    binary_map = {'Yes': 1, 'No': 0, 'High': 1, 'Medium': 0.5, 'Low': 0, 'None': 0}
    
    lifestyle_factors = ['Smoking', 'Alcohol Consumption', 'Exercise Habits', 'Stress Level']
    medical_factors = ['High Blood Pressure', 'High LDL Cholesterol', 'Low HDL Cholesterol', 'Diabetes']
    
    # Handle lifestyle factors
    lifestyle_columns = [col for col in lifestyle_factors if col in result.columns]
    if lifestyle_columns:
        # Convert categorical to numeric
        for col in lifestyle_columns:
            if result[col].dtype == 'object':
                result[f"{col}_numeric"] = result[col].map(lambda x: binary_map.get(x, 0) if isinstance(x, str) else x)
            else:
                result[f"{col}_numeric"] = result[col]
        
        # Calculate lifestyle risk score
        numeric_lifestyle_cols = [f"{col}_numeric" for col in lifestyle_columns]
        result['lifestyle_risk_score'] = result[numeric_lifestyle_cols].sum(axis=1)
        logger.info(f"Created lifestyle risk score from {len(lifestyle_columns)} factors")
    
    # Handle medical factors
    medical_columns = [col for col in medical_factors if col in result.columns]
    if medical_columns:
        # Convert categorical to numeric
        for col in medical_columns:
            if result[col].dtype == 'object':
                result[f"{col}_numeric"] = result[col].map(lambda x: binary_map.get(x, 0) if isinstance(x, str) else x)
            else:
                result[f"{col}_numeric"] = result[col]
        
        # Calculate medical risk score
        numeric_medical_cols = [f"{col}_numeric" for col in medical_columns]
        result['medical_risk_score'] = result[numeric_medical_cols].sum(axis=1)
        logger.info(f"Created medical risk score from {len(medical_columns)} factors")
    
    # Family history and age interaction
    if 'Family Heart Disease' in result.columns and 'Age' in result.columns:
        # Convert Family Heart Disease to numeric if needed
        if result['Family Heart Disease'].dtype == 'object':
            family_hd_numeric = result['Family Heart Disease'].map(binary_map)
        else:
            family_hd_numeric = result['Family Heart Disease']
        
        # Create age-family history interaction
        result['age_family_history_risk'] = result['Age'] * family_hd_numeric
        logger.info("Created age-family history interaction feature")
    
    return result

def select_features_with_model(X, y, estimator=None, threshold='mean', max_features=None):
    """
    Select features using a model-based approach.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : array-like
        Target values
    estimator : estimator object, default=None
        The base estimator to use for feature selection
    threshold : str or float, default='mean'
        Feature importance threshold
    max_features : int, default=None
        Maximum number of features to select
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    """
    logger.info("Selecting features with model-based approach")
    
    # Create estimator if not provided
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create feature selector
    selector = SelectFromModel(
        estimator=estimator,
        threshold=threshold,
        max_features=max_features
    )
    
    # Fit and transform
    try:
        X_selected = selector.fit_transform(X, y)
        
        # Get feature mask and names
        selected_features = X.columns[selector.get_support()]
        
        logger.info(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
        
        return X[selected_features]
    
    except Exception as e:
        logger.error(f"Feature selection failed: {e}")
        return X

def select_features_with_rfe(X, y, estimator=None, cv=5, step=1, min_features_to_select=5):
    """
    Select features using Recursive Feature Elimination with Cross-Validation.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : array-like
        Target values
    estimator : estimator object, default=None
        The base estimator to use for feature selection
    cv : int, default=5
        Cross-validation folds
    step : int or float, default=1
        Number or percentage of features to remove at each iteration
    min_features_to_select : int, default=5
        Minimum number of features to select
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features
    """
    print(min_features_to_select)
    min_features_to_select=5
    logger.info("Selecting features with RFE and cross-validation")
    
    # Create estimator if not provided
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create selector
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring='recall',
        min_features_to_select=min_features_to_select
    )
    
    # Fit and transform
    try:
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_]
        
        logger.info(f"RFE selected {len(selected_features)} features: {', '.join(selected_features)}")
        logger.info(f"Optimal number of features: {selector.n_features_}")
        
        return X[selected_features]
    
    except Exception as e:
        logger.error(f"RFE feature selection failed: {e}")
        return X
    
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
    
    # Only create interactions between numerical features
    numerical_features = enhanced_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Age and Blood Pressure interaction (if both exist as numerical features)
    if 'Age' in numerical_features and 'Blood Pressure' in numerical_features:
        enhanced_data['age_bp_interaction'] = enhanced_data['Age'] * enhanced_data['Blood Pressure'] / 100
        logger.info("Created Age x Blood Pressure interaction feature")
    
    # BMI and Cholesterol interaction (if both exist as numerical features)
    if 'BMI' in numerical_features and 'Cholesterol Level' in numerical_features:
        enhanced_data['bmi_cholesterol_interaction'] = enhanced_data['BMI'] * enhanced_data['Cholesterol Level'] / 100
        logger.info("Created BMI x Cholesterol interaction feature")
    
    # Track how many new features we've created
    new_features = len(enhanced_data.columns) - len(data.columns)
    logger.info(f"Created {new_features} new interaction features")
    
    return enhanced_data