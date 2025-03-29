"""
Random Forest model for heart disease prediction.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger(__name__)

def create_random_forest_model(preprocessor, **kwargs):
    """
    Create a random forest pipeline with preprocessing.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    **kwargs : dict
        Additional parameters for RandomForestClassifier
        
    Returns:
    --------
    Pipeline
        Pipeline with preprocessing and random forest
    """
    logger.info("Creating random forest pipeline")
    
    # Create pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**kwargs))
    ])
    
    return rf_pipeline

def get_feature_importance(model, feature_names, X, y, importance_type='gini'):
    """
    Get feature importance from the random forest model.
    
    Parameters:
    -----------
    model : Pipeline
        Fitted random forest pipeline
    feature_names : list
        Names of the features
    X : array-like
        Input features for permutation importance
    y : array-like
        Target values for permutation importance
    importance_type : str, default='gini'
        Type of feature importance ('gini' or 'permutation')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances
    """
    logger.info(f"Calculating {importance_type} feature importance")
    
    if importance_type == 'gini':
        # Get built-in feature importance (Gini importance)
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        })
        
    else:
        raise ValueError(f"Unknown importance type: {importance_type}")
        
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    logger.info(f"Top 5 most important features: {', '.join(importance_df['Feature'].head(5).tolist())}")
    
    return importance_df

def analyze_tree_paths(model, feature_names):
    """
    Analyze decision paths in the random forest to gain insights.
    
    Parameters:
    -----------
    model : Pipeline
        Fitted random forest pipeline
    feature_names : list
        Names of the features
        
    Returns:
    --------
    dict
        Dictionary with tree path analysis
    """
    logger.info("Analyzing decision tree paths in random forest")
    
    # Extract random forest classifier
    rf = model.named_steps['classifier']
    
    # Analyze feature co-occurrence in decision paths
    feature_counts = {}
    feature_depths = {}
    
    # Initialize counts
    for feature in feature_names:
        feature_counts[feature] = 0
        feature_depths[feature] = []
    
    # Analyze trees
    for i, tree in enumerate(rf.estimators_):
        # Get decision path nodes
        n_nodes = tree.tree_.node_count
        feature_used = tree.tree_.feature
        
        # Count feature usage
        for node_id in range(n_nodes):
            feature_idx = feature_used[node_id]
            if feature_idx != -2:  # -2 indicates leaf node
                feature_name = feature_names[feature_idx]
                feature_counts[feature_name] += 1
                # Store depth in tree (node_id is a proxy for depth in simple analysis)
                feature_depths[feature_name].append(node_id)
    
    # Calculate average depth for each feature
    avg_depths = {
        feature: np.mean(depths) if depths else 0 
        for feature, depths in feature_depths.items()
    }
    
    # Prepare results
    results = {
        'feature_usage_counts': feature_counts,
        'feature_avg_depth': avg_depths,
        'top_features_by_usage': sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        'top_features_by_depth': sorted(avg_depths.items(), key=lambda x: x[1])[:10]
    }
    
    logger.info("Decision path analysis completed")
    
    return results
