"""
Visualization functions for heart disease prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set the default plotting style."""
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

def plot_feature_distributions(data, target_column):
    """
    Plot distributions of features by target class.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    target_column : str
        Name of the target column
    """
    logger.info("Plotting feature distributions")
    
    set_plotting_style()
    
    # Separate numerical and categorical features
    numerical_features = data.select_dtypes(include=np.number).columns.tolist()
    categorical_features = data.select_dtypes(exclude=np.number).columns.tolist()
    
    # Remove target from features if present
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Plot numerical features
    if numerical_features:
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                ax = axes[i]
                
                # Check for target as numerical or categorical
                if data[target_column].dtype == 'object':
                    # Plot histograms by category
                    for category in data[target_column].unique():
                        subset = data[data[target_column] == category]
                        sns.histplot(subset[feature], ax=ax, label=category, alpha=0.5, kde=True)
                    ax.legend(title=target_column)
                else:
                    # Plot histograms by binary target
                    sns.histplot(data[data[target_column] == 0][feature], ax=ax, 
                                label=f"{target_column}=0", alpha=0.5, kde=True)
                    sns.histplot(data[data[target_column] == 1][feature], ax=ax, 
                                label=f"{target_column}=1", alpha=0.5, kde=True)
                    ax.legend()
                
                ax.set_title(f"Distribution of {feature}")
                ax.set_xlabel(feature)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        
    # Plot categorical features
    if categorical_features:
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])  # Ensure axes is indexable
        axes = axes.flatten()
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes):
                ax = axes[i]
                
                # Create a cross-tabulation
                counts = pd.crosstab(data[feature], data[target_column])
                percentages = counts.div(counts.sum(axis=1), axis=0)
                
                # Plot stacked bar chart
                percentages.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f"{feature} vs {target_column}")
                ax.set_ylabel("Percentage")
                ax.set_ylim(0, 1.0)
                
                # Add count annotations
                for container in ax.containers:
                    ax.bar_label(container, label_type='center', fmt='%.2f')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()

def plot_correlation_heatmap(data, numeric_only=True):
    """
    Plot correlation heatmap for numerical features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    numeric_only : bool, default=True
        Whether to include only numeric columns
    """
    logger.info("Plotting correlation heatmap")
    
    set_plotting_style()
    
    # Filter numeric columns if required
    if numeric_only:
        data = data.select_dtypes(include=np.number)
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap="coolwarm", vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

def plot_feature_importance(importance_df, title="Feature Importance"):
    """
    Plot feature importance bar chart.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with columns 'Feature' and one of ['Importance', 'Coefficient', 'Absolute_Value']
    title : str, default="Feature Importance"
        Title for the plot
    """
    logger.info(f"Plotting {title}")
    
    set_plotting_style()
    
    # Determine which column to use for importance values
    importance_column = None
    for col in ['Importance', 'Coefficient', 'Absolute_Value']:
        if col in importance_df.columns:
            importance_column = col
            break
    
    if importance_column is None:
        logger.error(f"No recognized importance column found in DataFrame. Columns: {importance_df.columns.tolist()}")
        importance_column = importance_df.columns[1] if len(importance_df.columns) > 1 else "value"
        logger.warning(f"Using column '{importance_column}' as fallback for importance values")
    
    # Sort by absolute importance if needed
    if importance_column == 'Coefficient' and 'Absolute_Value' in importance_df.columns:
        importance_df = importance_df.sort_values('Absolute_Value', ascending=False)
    else:
        importance_df = importance_df.sort_values(importance_column, ascending=False)
    
    # Take top N features for readability
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    
    # Create the bar chart
    bars = plt.barh(top_features['Feature'], top_features[importance_column])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center')
    
    plt.xlabel(importance_column)
    plt.title(title)
    plt.tight_layout()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    class_names : list, optional
        Names for the classes
    """
    logger.info("Plotting confusion matrix")
    
    set_plotting_style()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set class names if not provided
    if class_names is None:
        class_names = ["No Disease", "Heart Disease"]
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count plot
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Percentage plot
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()

def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_proba : array-like
        Predicted probabilities for the positive class
    model_name : str, default="Model"
        Name of the model
    """
    logger.info(f"Plotting ROC curve for {model_name}")
    
    set_plotting_style()
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

def plot_precision_recall_curve(y_true, y_proba, model_name="Model"):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_proba : array-like
        Predicted probabilities for the positive class
    model_name : str, default="Model"
        Name of the model
    """
    logger.info(f"Plotting precision-recall curve for {model_name}")
    
    set_plotting_style()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name}')
    
    # Add no-skill line (baseline)
    no_skill = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', label='No Skill')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

def plot_model_comparison(comparison_df):
    """
    Plot model comparison bar chart.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with model comparison metrics
    """
    logger.info("Plotting model comparison")
    
    set_plotting_style()
    
    # Select metrics to plot
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1']
    if 'ROC AUC' in comparison_df.columns:
        metrics.append('ROC AUC')
    
    # Prepare data for plotting
    plot_data = comparison_df.melt(id_vars=['Model'], 
                                   value_vars=metrics, 
                                   var_name='Metric', 
                                   value_name='Value')
    
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='Model', y='Value', hue='Metric', data=plot_data)
    
    plt.title('Model Comparison')
    plt.ylim(0, 1.1)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Add value labels
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f')