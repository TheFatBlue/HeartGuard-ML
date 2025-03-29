#!/usr/bin/env python
"""
Main script for the heart disease prediction project.
This script runs the full pipeline from data loading to model evaluation.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Import project modules
from src.config import (
    ROOT_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, CV_FOLDS,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, SMOTE_PARAMS
)
from src.data.data_loader import load_data, split_features_target, save_processed_data
from src.data.preprocessor import (
    check_data_quality, clean_data, create_preprocessing_pipeline, encode_target
)
from src.features.feature_engineering import (
    create_interaction_features, compute_lipid_ratio, get_feature_names
)
from src.models.baseline import MajorityClassifier
from src.models.logistic import create_logistic_model, analyze_coefficients
from src.models.random_forest import create_random_forest_model, get_feature_importance, analyze_tree_paths
from src.utils.imbalance import apply_smote, get_class_weights, create_cost_sensitive_sample_weights
from src.utils.metrics import evaluate_classifier, find_optimal_threshold, compare_models, analyze_misclassifications
from src.visualization.visualize import (
    plot_feature_distributions, plot_correlation_heatmap, plot_feature_importance,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_model_comparison
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, 'heart_disease_prediction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heart Disease Prediction Pipeline')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the raw data CSV (default: use from config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: use from config)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--use_smote', action='store_true',
                        help='Use SMOTE for handling class imbalance')
    parser.add_argument('--skip_eda', action='store_true',
                        help='Skip exploratory data analysis visualizations')
    
    return parser.parse_args()

def main():
    """Run the heart disease prediction pipeline."""
    args = parse_arguments()
    
    # Set up paths
    data_path = args.data_path or RAW_DATA_PATH
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting heart disease prediction pipeline")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Load and explore data
    logger.info("STEP 1: Data loading and exploration")
    data = load_data(data_path)
    
    # Check data quality
    quality_summary = check_data_quality(data)
    
    # Step 2: Data preprocessing
    logger.info("STEP 2: Data preprocessing")
    cleaned_data = clean_data(data)
    
    # Save processed data
    save_processed_data(cleaned_data, PROCESSED_DATA_PATH)
    
    # Exploratory Data Analysis
    if not args.skip_eda:
        logger.info("Generating exploratory data analysis plots")
        plot_feature_distributions(cleaned_data, TARGET_COLUMN)
        plt.savefig(output_dir / 'feature_distributions.png')
        
        plot_correlation_heatmap(cleaned_data)
        plt.savefig(output_dir / 'correlation_heatmap.png')
    
    # Step 3: Feature engineering
    logger.info("STEP 3: Feature engineering")
    
    # Create additional features
    enhanced_data = create_interaction_features(cleaned_data)
    enhanced_data = compute_lipid_ratio(enhanced_data)
    
    # Split features and target
    X, y = split_features_target(enhanced_data, TARGET_COLUMN)
    
    # Create a preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    
    # Step 4: Train-test split
    logger.info("STEP 4: Train-test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    logger.info(f"Training set class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test set class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Handle class imbalance if requested
    if args.use_smote:
        logger.info("Applying SMOTE to handle class imbalance")
        # First apply preprocessing
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        # Then apply SMOTE to the preprocessed features
        logger.info(f"Before SMOTE - X_train_preprocessed shape: {X_train_preprocessed.shape}")
        logger.info(f"Before SMOTE - y_train shape: {y_train.shape}")
        
        X_train_resampled, y_train_resampled = apply_smote(X_train_preprocessed, y_train, **SMOTE_PARAMS)
        
        logger.info(f"After SMOTE - X_train_resampled shape: {X_train_resampled.shape}")
        logger.info(f"After SMOTE - y_train_resampled shape: {y_train_resampled.shape}")
        logger.info(f"After SMOTE - Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    else:
        # If not using SMOTE, we'll handle class imbalance through class weights
        # and implement preprocessing within the model pipelines
        X_train_resampled, y_train_resampled = X_train, y_train
        X_train_preprocessed, X_test_preprocessed = None, None
    
    # Step 5: Train and evaluate models
    logger.info("STEP 5: Model training and evaluation")
    
    # Get feature names after preprocessing (needed for interpretation)
    # Ensure preprocessor is fitted
    try:
        if X_train_preprocessed is None:
            logger.info("Fitting preprocessor to get feature names")
            preprocessor.fit(X_train)
        
        # Get transformed feature names
        logger.info("Extracting feature names from preprocessor")
        feature_names = get_feature_names(preprocessor)
        logger.info(f"Extracted {len(feature_names)} feature names")
        
        # If we don't have feature names or they're incorrect length, use generic names
        if not feature_names or (X_train_preprocessed is not None and len(feature_names) != X_train_preprocessed.shape[1]):
            logger.warning("Feature names don't match preprocessed data shape, using generic names")
            feature_names = [f"feature_{i}" for i in range(X_train_preprocessed.shape[1] if X_train_preprocessed is not None else preprocessor.transform(X_train.iloc[:1]).shape[1])]
    except Exception as e:
        logger.error(f"Error getting feature names: {e}")
        # Fallback to generic feature names
        feature_names = [f"feature_{i}" for i in range(X_train_preprocessed.shape[1] if X_train_preprocessed is not None else 20)]
        logger.info(f"Using {len(feature_names)} generic feature names")
    
    # Dictionary to store model evaluation results
    model_results = {}
    
    # 5.1: Baseline (Majority Classifier)
    logger.info("Training Baseline (Majority Classifier)")
    baseline = MajorityClassifier()
    
    if X_train_preprocessed is not None:
        baseline.fit(X_train_preprocessed, y_train_resampled)
        y_pred_baseline = baseline.predict(X_test_preprocessed)
    else:
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
    
    # Evaluate baseline
    baseline_metrics = evaluate_classifier(y_test, y_pred_baseline)
    model_results['Baseline'] = baseline_metrics
    
    # 5.2: Logistic Regression
    logger.info("Training Logistic Regression")
    # Create and train logistic regression model
    if X_train_preprocessed is not None and args.use_smote:
        # If we've already applied SMOTE and preprocessing
        from sklearn.linear_model import LogisticRegression
        
        # Make sure we're using the resampled data from SMOTE correctly
        logger.info(f"Training logistic regression with SMOTE resampled data: X shape: {X_train_resampled.shape}, y shape: {y_train_resampled.shape}")
        
        logistic_model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        logistic_model.fit(X_train_resampled, y_train_resampled)
        y_pred_logistic = logistic_model.predict(X_test_preprocessed)
        y_proba_logistic = logistic_model.predict_proba(X_test_preprocessed)[:, 1]
    else:
        # Use pipeline with preprocessing
        logger.info("Training logistic regression with pipeline")
        logistic_model = create_logistic_model(preprocessor, **LOGISTIC_REGRESSION_PARAMS)
        logistic_model.fit(X_train, y_train)
        y_pred_logistic = logistic_model.predict(X_test)
        y_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]
    
    # Evaluate logistic regression
    logistic_metrics = evaluate_classifier(y_test, y_pred_logistic, y_proba_logistic)
    model_results['Logistic Regression'] = logistic_metrics
    
    # Analyze logistic regression coefficients
    if X_train_preprocessed is not None and args.use_smote:
        # Direct model
        logger.info("Extracting coefficients from logistic regression model")
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': logistic_model.coef_[0],
            'Absolute_Value': np.abs(logistic_model.coef_[0])
        }).sort_values('Absolute_Value', ascending=False)
    else:
        # Pipeline model
        logger.info("Analyzing coefficients from logistic regression pipeline")
        coef_df = analyze_coefficients(logistic_model, feature_names)
    
    # Plot feature importance
    plot_feature_importance(coef_df, "Logistic Regression - Feature Importance")
    plt.savefig(output_dir / 'logistic_regression_importance.png')
    plt.close() # Close the plot to avoid memory issues
    
    # 5.3: Random Forest
    logger.info("Training Random Forest")
    # Create and train random forest model
    if X_train_preprocessed is not None and args.use_smote:
        # If we've already applied SMOTE and preprocessing
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info(f"Training random forest with SMOTE resampled data: X shape: {X_train_resampled.shape}, y shape: {y_train_resampled.shape}")
        
        rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        rf_model.fit(X_train_resampled, y_train_resampled)
        y_pred_rf = rf_model.predict(X_test_preprocessed)
        y_proba_rf = rf_model.predict_proba(X_test_preprocessed)[:, 1]
    else:
        # Use pipeline with preprocessing
        logger.info("Training random forest with pipeline")
        rf_model = create_random_forest_model(preprocessor, **RANDOM_FOREST_PARAMS)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate random forest
    rf_metrics = evaluate_classifier(y_test, y_pred_rf, y_proba_rf)
    model_results['Random Forest'] = rf_metrics
    
    # Get feature importance for random forest
    if X_train_preprocessed is not None and args.use_smote:
        # Direct model
        logger.info("Extracting feature importance from random forest model")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        # Pipeline model
        logger.info("Analyzing feature importance from random forest pipeline")
        importance_df = get_feature_importance(rf_model, feature_names, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(importance_df, "Random Forest - Feature Importance")
    plt.savefig(output_dir / 'random_forest_importance.png')
    plt.close() # Close the plot to avoid memory issues
    
    # Step 6: Model comparison and interpretation
    logger.info("STEP 6: Model comparison and interpretation")
    
    # Compare models
    comparison_df = compare_models(model_results, target_metric='recall')
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # Plot model comparison
    plot_model_comparison(comparison_df)
    plt.savefig(output_dir / 'model_comparison.png')
    plt.close()
    
    # Plot confusion matrices
    plot_confusion_matrix(y_test, y_pred_logistic, ["No Disease", "Heart Disease"])
    plt.savefig(output_dir / 'logistic_confusion_matrix.png')
    plt.close()
    
    plot_confusion_matrix(y_test, y_pred_rf, ["No Disease", "Heart Disease"])
    plt.savefig(output_dir / 'random_forest_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curves
    plot_roc_curve(y_test, y_proba_logistic, "Logistic Regression")
    plt.savefig(output_dir / 'logistic_roc_curve.png')
    plt.close()
    
    plot_roc_curve(y_test, y_proba_rf, "Random Forest")
    plt.savefig(output_dir / 'random_forest_roc_curve.png')
    plt.close()
    
    # Plot precision-recall curves
    plot_precision_recall_curve(y_test, y_proba_logistic, "Logistic Regression")
    plt.savefig(output_dir / 'logistic_pr_curve.png')
    plt.close()
    
    plot_precision_recall_curve(y_test, y_proba_rf, "Random Forest")
    plt.savefig(output_dir / 'random_forest_pr_curve.png')
    plt.close()
    
    # Step 7: Find optimal threshold to maximize recall while maintaining precision
    logger.info("STEP 7: Optimizing decision threshold")
    
    # Find optimal threshold for each model
    logistic_threshold = find_optimal_threshold(y_test, y_proba_logistic, optimize_for='recall', min_precision=0.5)
    rf_threshold = find_optimal_threshold(y_test, y_proba_rf, optimize_for='recall', min_precision=0.5)
    
    # Apply optimal thresholds
    y_pred_logistic_optimized = (y_proba_logistic >= logistic_threshold).astype(int)
    y_pred_rf_optimized = (y_proba_rf >= rf_threshold).astype(int)
    
    # Evaluate with optimized thresholds
    logistic_optimized_metrics = evaluate_classifier(y_test, y_pred_logistic_optimized, y_proba_logistic)
    rf_optimized_metrics = evaluate_classifier(y_test, y_pred_rf_optimized, y_proba_rf)
    
    model_results['Logistic Regression (Optimized)'] = logistic_optimized_metrics
    model_results['Random Forest (Optimized)'] = rf_optimized_metrics
    
    # Create final comparison
    final_comparison = compare_models(model_results, target_metric='recall')
    final_comparison.to_csv(output_dir / 'final_model_comparison.csv', index=False)
    
    # Plot final model comparison
    plot_model_comparison(final_comparison)
    plt.savefig(output_dir / 'final_model_comparison.png')
    plt.close()
    
    # Step 8: Analyze misclassifications for best model
    logger.info("STEP 8: Analyzing misclassifications")
    
    # Find best model based on recall
    best_model_name = final_comparison.iloc[0]['Model']
    logger.info(f"Best model based on recall: {best_model_name}")
    
    # Get predictions from best model
    if best_model_name.startswith('Logistic'):
        best_pred = y_pred_logistic_optimized if 'Optimized' in best_model_name else y_pred_logistic
    elif best_model_name.startswith('Random Forest'):
        best_pred = y_pred_rf_optimized if 'Optimized' in best_model_name else y_pred_rf
    else:
        best_pred = y_pred_baseline
    
    # Analyze misclassifications
    misclassification_analysis = analyze_misclassifications(X_test, y_test, best_pred, X_test.columns)
    
    # Step 9: Save results
    logger.info("STEP 9: Saving final results")
    
    # Save best model coefficients/feature importance
    if best_model_name.startswith('Logistic'):
        coef_df.to_csv(output_dir / 'best_model_coefficients.csv', index=False)
    elif best_model_name.startswith('Random Forest'):
        importance_df.to_csv(output_dir / 'best_model_importance.csv', index=False)
    
    # Summary of false negatives (critical errors)
    if 'false_negative_stats' in misclassification_analysis:
        fn_stats = misclassification_analysis['false_negative_stats']
        fn_stats.to_csv(output_dir / 'false_negative_statistics.csv')
    
    logger.info("Heart disease prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()