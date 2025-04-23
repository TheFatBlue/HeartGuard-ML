# src/models/xgboost.py
"""
XGBoost model for heart disease prediction.
"""
import xgboost as xgb
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_xgboost_model(preprocessor, **kwargs):
    """
    Create an XGBoost pipeline with preprocessing.

    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    **kwargs : dict
        Additional parameters for XGBoostClassifier

    Returns:
    --------
    Pipeline
        Pipeline with preprocessing and XGBoost classifier
    """
    logger.info("Creating XGBoost pipeline")

    # Extract early stopping rounds if provided, else default
    fit_params = {}
    if 'early_stopping_rounds' in kwargs:
        fit_params['classifier__early_stopping_rounds'] = kwargs.pop('early_stopping_rounds')
        # Need an eval_set for early stopping, usually provided during fit
        # fit_params['classifier__eval_set'] = [(X_val, y_val)] # Requires validation set passed to fit

    # Ensure objective and eval_metric are set if not provided
    if 'objective' not in kwargs:
        kwargs['objective'] = 'binary:logistic'
    if 'eval_metric' not in kwargs:
        kwargs['eval_metric'] = 'logloss' # Default, consider 'aucpr' for recall focus
    if 'use_label_encoder' not in kwargs:
         kwargs['use_label_encoder'] = False # Suppress warning

    # Create pipeline
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**kwargs))
    ])

    return xgb_pipeline, fit_params

def get_xgboost_feature_importance(model, feature_names):
    """
    Get feature importance from the XGBoost model.

    Parameters:
    -----------
    model : Pipeline
        Fitted XGBoost pipeline
    feature_names : list
        Names of the features after preprocessing

    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances (using default 'weight' type)
    """
    logger.info("Calculating XGBoost feature importance")

    try:
        importances = model.named_steps['classifier'].feature_importances_

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)

        logger.info(f"Top 5 most important features (XGBoost): {', '.join(importance_df['Feature'].head(5).tolist())}")

        return importance_df
    except Exception as e:
        logger.error(f"Error getting XGBoost feature importance: {e}")
        return pd.DataFrame({'Feature': feature_names, 'Importance': 0})
