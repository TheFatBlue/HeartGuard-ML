"""
Logistic regression model for heart disease prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_logistic_model(preprocessor, **kwargs):
    """
    Create a logistic regression pipeline with preprocessing.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    **kwargs : dict
        Additional parameters for LogisticRegression
        
    Returns:
    --------
    Pipeline
        Pipeline with preprocessing and logistic regression
    """
    logger.info("Creating logistic regression pipeline")
    
    # Create pipeline
    logistic_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**kwargs))
    ])
    
    return logistic_pipeline

def analyze_coefficients(model, feature_names):
    """
    Analyze the coefficients of a fitted logistic regression model.
    
    Parameters:
    -----------
    model : Pipeline
        Fitted logistic regression pipeline
    feature_names : list
        Names of the features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances
    """
    logger.info("Analyzing logistic regression coefficients")
    
    # Extract coefficients
    coefficients = model.named_steps['classifier'].coef_[0]
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Value': np.abs(coefficients)
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('Absolute_Value', ascending=False)
    
    logger.info(f"Top 5 most important features: {', '.join(coef_df['Feature'].head(5).tolist())}")
    
    return coef_df

def interpret_logistic_predictions(X, model, threshold=0.5):
    """
    Interpret predictions from the logistic regression model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features
    model : Pipeline
        Fitted logistic regression pipeline
    threshold : float, default=0.5
        Probability threshold for positive class
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with prediction probabilities and interpretations
    """
    logger.info(f"Interpreting logistic regression predictions with threshold {threshold}")
    
    # Get probabilities
    probas = model.predict_proba(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Probability': probas[:, 1],
        'Prediction': (probas[:, 1] >= threshold).astype(int)
    })
    
    # Add interpretation
    results['Risk_Level'] = pd.cut(
        results['Probability'], 
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    )
    
    # Count risk levels
    risk_counts = results['Risk_Level'].value_counts()
    logger.info(f"Risk level distribution: {risk_counts.to_dict()}")
    
    return results