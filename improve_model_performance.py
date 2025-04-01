#!/usr/bin/env python
"""
Script to improve heart disease prediction model performance.
This script implements advanced techniques beyond the basic pipeline.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Import project modules
from src.config import (
    ROOT_DIR, RAW_DATA_PATH, TARGET_COLUMN, CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES, RANDOM_STATE
)
from src.data.data_loader import load_data, split_features_target
from src.data.preprocessor import clean_data, create_preprocessing_pipeline, encode_target
from src.features.feature_engineering import create_interaction_features, compute_lipid_ratio
from src.features.advanced_features import (
    MedicalRiskScoreTransformer, create_polynomial_features, 
    create_risk_clusters, select_features_with_model, select_features_with_rfe
)
from src.models.hyperparameter_tuning import (
    tune_logistic_regression, tune_random_forest, 
    tune_xgboost, tune_gradient_boosting
)
from src.models.ensemble import (
    create_voting_ensemble, create_stacking_ensemble, 
    create_recall_optimized_ensemble, calibrate_model_threshold
)
from src.utils.imbalance import apply_smote, apply_undersampling
from src.utils.metrics import evaluate_classifier, compare_models
from src.visualization.visualize import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_model_comparison
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, 'heart_disease_improved.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heart Disease Prediction - Model Improvement')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the raw data CSV (default: use from config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: use from config)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all improvement techniques')
    parser.add_argument('--advanced_features', action='store_true',
                        help='Use advanced feature engineering')
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--build_ensemble', action='store_true',
                        help='Build ensemble models')
    parser.add_argument('--calibrate_thresholds', action='store_true',
                        help='Calibrate classification thresholds')
    
    return parser.parse_args()

def main():
    """Run the model improvement pipeline."""
    args = parse_arguments()
    
    # Set up paths
    data_path = args.data_path or RAW_DATA_PATH
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / 'output' / 'improved'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting heart disease prediction model improvement")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # If run_all is specified, enable all improvement techniques
    if args.run_all:
        args.advanced_features = True
        args.hyperparameter_tuning = True
        args.build_ensemble = True
        args.calibrate_thresholds = True
    
    # Step 1: Load and preprocess data
    logger.info("STEP 1: Loading and preprocessing data")
    
    # Load data
    data = load_data(data_path)
    cleaned_data = clean_data(data)
    
    # Split features and target
    X, y = split_features_target(cleaned_data, TARGET_COLUMN)
    
    # Encode target if needed
    y = encode_target(y)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Step 2: Feature Engineering
    logger.info("STEP 2: Feature Engineering")
    
    # Basic feature engineering
    X_train = create_interaction_features(X_train)
    X_train = compute_lipid_ratio(X_train)
    
    # Apply same transformations to validation and test sets
    X_val = create_interaction_features(X_val)
    X_val = compute_lipid_ratio(X_val)
    
    X_test = create_interaction_features(X_test)
    X_test = compute_lipid_ratio(X_test)
    
    # Advanced feature engineering if requested
    if args.advanced_features:
        logger.info("Applying advanced feature engineering")
        
        # Medical risk scores
        medical_transformer = MedicalRiskScoreTransformer()
        X_train = medical_transformer.transform(X_train)
        X_val = medical_transformer.transform(X_val)
        X_test = medical_transformer.transform(X_test)
        
        # Risk clusters
        X_train = create_risk_clusters(X_train)
        X_val = create_risk_clusters(X_val)
        X_test = create_risk_clusters(X_test)
        
        # Polynomial features (only for select numerical features to avoid explosion)
        important_num_features = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI']
        numerical_subset = [col for col in important_num_features if col in X_train.columns]
        
        if numerical_subset:
            # Create polynomial features on subset
            poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
            # Fit and transform train set
            poly_features_train = poly_transformer.fit_transform(X_train[numerical_subset])
            
            # Transform validation and test sets
            poly_features_val = poly_transformer.transform(X_val[numerical_subset])
            poly_features_test = poly_transformer.transform(X_test[numerical_subset])
            
            # Get feature names
            if hasattr(poly_transformer, 'get_feature_names_out'):
                poly_feature_names = poly_transformer.get_feature_names_out(numerical_subset)
            else:
                poly_feature_names = poly_transformer.get_feature_names(numerical_subset)
            
            # Create DataFrames
            poly_df_train = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=X_train.index)
            poly_df_val = pd.DataFrame(poly_features_val, columns=poly_feature_names, index=X_val.index)
            poly_df_test = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=X_test.index)
            
            # Remove original features from polynomial DataFrames
            for feature in numerical_subset:
                if feature in poly_df_train.columns:
                    poly_df_train = poly_df_train.drop(columns=[feature])
                    poly_df_val = poly_df_val.drop(columns=[feature])
                    poly_df_test = poly_df_test.drop(columns=[feature])
            
            # Merge with original DataFrames
            X_train = pd.concat([X_train, poly_df_train], axis=1)
            X_val = pd.concat([X_val, poly_df_val], axis=1)
            X_test = pd.concat([X_test, poly_df_test], axis=1)
            
            logger.info(f"Added {poly_df_train.shape[1]} polynomial features")
    
    # Step 3: Handle Class Imbalance
    logger.info("STEP 3: Handling class imbalance with SMOTE")
    
    # Apply SMOTE to the training data
    X_train_resampled, y_train_resampled = apply_smote(
        X_train.values, y_train, random_state=RANDOM_STATE
    )
    
    # Convert back to DataFrame
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    
    logger.info(f"After SMOTE - Training set size: {X_train_resampled.shape[0]} samples")
    logger.info(f"Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    # Step 4: Feature Selection
    logger.info("STEP 4: Feature selection")
    
    # We'll use RFE for feature selection
    X_train_selected = select_features_with_rfe(
        X_train_resampled, y_train_resampled, 
        min_features_to_select=max(10, X_train.shape[1] // 3)
    )
    
    # Get selected features
    selected_features = X_train_selected.columns.tolist()
    
    # Apply same feature selection to validation and test sets
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    logger.info(f"Selected {len(selected_features)} features")
    
    # Step 5: Model Training and Hyperparameter Tuning
    logger.info("STEP 5: Model Training and Hyperparameter Tuning")
    
    # Dictionary to store models
    models = {}
    
    # Base models without hyperparameter tuning
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    # Add base models
    models['Logistic Regression'] = LogisticRegression(
        class_weight='balanced', random_state=RANDOM_STATE
    )
    
    models['Random Forest'] = RandomForestClassifier(
        class_weight='balanced', random_state=RANDOM_STATE
    )
    
    models['XGBoost'] = XGBClassifier(
        scale_pos_weight=4, # Ratio of negative to positive samples
        random_state=RANDOM_STATE
    )
    
    models['Gradient Boosting'] = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )
    
    # Neural Network
    models['Neural Network'] = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=RANDOM_STATE
    )
    
    # Hyperparameter tuning if requested
    if args.hyperparameter_tuning:
        logger.info("Performing hyperparameter tuning")
        
        # Tune logistic regression
        models['Tuned Logistic Regression'] = tune_logistic_regression(
            X_train_selected, y_train_resampled, cv=3
        )
        
        # Tune random forest (using fewer iterations to save time)
        models['Tuned Random Forest'] = tune_random_forest(
            X_train_selected, y_train_resampled, cv=3
        )
        
        # Tune XGBoost
        models['Tuned XGBoost'] = tune_xgboost(
            X_train_selected, y_train_resampled, cv=3
        )
        
        # Tune Gradient Boosting
        models['Tuned Gradient Boosting'] = tune_gradient_boosting(
            X_train_selected, y_train_resampled, cv=3
        )
    
    # Fit all models
    logger.info("Fitting individual models")
    
    # Dictionary to store results
    model_results = {}
    
    for name, model in models.items():
        logger.info(f"Fitting {name}")
        model.fit(X_train_selected, y_train_resampled)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val_selected)
        y_proba = model.predict_proba(X_val_selected)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Store evaluation results
        val_metrics = evaluate_classifier(y_val, y_pred, y_proba)
        model_results[name] = val_metrics
        
        logger.info(f"{name} - Validation recall: {val_metrics['recall']:.4f}, "
                   f"precision: {val_metrics['precision']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
    
    # Step 6: Ensemble Models
    if args.build_ensemble:
        logger.info("STEP 6: Building ensemble models")
        
        # Prepare estimators for ensemble
        estimators = [(name, model) for name, model in models.items()]
        
        # Voting ensemble
        voting_ensemble = create_voting_ensemble(estimators, voting='soft')
        voting_ensemble.fit(X_train_selected, y_train_resampled)
        
        # Stacking ensemble
        stacking_ensemble = create_stacking_ensemble(estimators)
        stacking_ensemble.fit(X_train_selected, y_train_resampled)
        
        # Recall-optimized ensemble
        recall_ensemble = create_recall_optimized_ensemble(
            estimators, X_train_selected, y_train_resampled
        )
        recall_ensemble.fit(X_train_selected, y_train_resampled)
        
        # Add ensembles to models dictionary
        models['Voting Ensemble'] = voting_ensemble
        models['Stacking Ensemble'] = stacking_ensemble
        models['Recall Ensemble'] = recall_ensemble
        
        # Evaluate ensembles on validation set
        for name in ['Voting Ensemble', 'Stacking Ensemble', 'Recall Ensemble']:
            model = models[name]
            y_pred = model.predict(X_val_selected)
            y_proba = model.predict_proba(X_val_selected)[:, 1]
            
            val_metrics = evaluate_classifier(y_val, y_pred, y_proba)
            model_results[name] = val_metrics
            
            logger.info(f"{name} - Validation recall: {val_metrics['recall']:.4f}, "
                       f"precision: {val_metrics['precision']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
    
    # Step 7: Threshold Calibration
    if args.calibrate_thresholds:
        logger.info("STEP 7: Calibrating classification thresholds")
        
        # Models with calibrated thresholds
        calibrated_models = {}
        calibrated_results = {}
        
        # Calibrate thresholds for all models with predict_proba
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                logger.info(f"Calibrating threshold for {name}")
                
                # Find optimal threshold
                threshold = calibrate_model_threshold(
                    model, X_val_selected, y_val, 
                    optimize_for='recall', min_precision=0.15
                )
                
                # Store model and threshold
                calibrated_models[f"{name} (Calibrated)"] = (model, threshold)
                
                # Evaluate with calibrated threshold
                y_proba = model.predict_proba(X_val_selected)[:, 1]
                y_pred_calibrated = (y_proba >= threshold).astype(int)
                
                cal_metrics = evaluate_classifier(y_val, y_pred_calibrated, y_proba)
                calibrated_results[f"{name} (Calibrated)"] = cal_metrics
                
                logger.info(f"{name} (Calibrated) - Validation recall: {cal_metrics['recall']:.4f}, "
                           f"precision: {cal_metrics['precision']:.4f}, "
                           f"F1: {cal_metrics['f1']:.4f}")
        
        # Add calibrated results to model_results
        model_results.update(calibrated_results)
    
    # Step 8: Final Evaluation
    logger.info("STEP 8: Final evaluation on test set")
    
    # Compare model results on validation set
    validation_comparison = compare_models(model_results, target_metric='recall')
    validation_comparison.to_csv(output_dir / 'validation_model_comparison.csv', index=False)
    
    # Plot model comparison
    plot_model_comparison(validation_comparison)
    plt.savefig(output_dir / 'validation_model_comparison.png')
    plt.close()
    
    # Select the best model based on validation recall
    best_model_name = validation_comparison.iloc[0]['Model']
    logger.info(f"Best model based on validation recall: {best_model_name}")
    
    # Special handling for calibrated models
    if '(Calibrated)' in best_model_name and args.calibrate_thresholds:
        base_name = best_model_name.replace(' (Calibrated)', '')
        best_model = models[base_name]
        best_threshold = calibrated_models[best_model_name][1]
        
        # Evaluate on test set with calibrated threshold
        y_proba_test = best_model.predict_proba(X_test_selected)[:, 1]
        y_pred_test = (y_proba_test >= best_threshold).astype(int)
        
        logger.info(f"Evaluating {best_model_name} on test set with threshold {best_threshold:.4f}")
    else:
        # Use regular model without calibration
        best_model = models[best_model_name]
        
        # Evaluate on test set
        y_pred_test = best_model.predict(X_test_selected)
        y_proba_test = best_model.predict_proba(X_test_selected)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        logger.info(f"Evaluating {best_model_name} on test set")
    
    # Calculate test metrics
    test_metrics = evaluate_classifier(y_test, y_pred_test, y_proba_test)
    
    logger.info(f"Test performance - Accuracy: {test_metrics['accuracy']:.4f}, "
               f"Recall: {test_metrics['recall']:.4f}, "
               f"Precision: {test_metrics['precision']:.4f}, "
               f"F1: {test_metrics['f1']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_test, ["No Disease", "Heart Disease"])
    plt.savefig(output_dir / 'best_model_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve if probability predictions are available
    if y_proba_test is not None:
        plot_roc_curve(y_test, y_proba_test, best_model_name)
        plt.savefig(output_dir / 'best_model_roc_curve.png')
        plt.close()
        
        plot_precision_recall_curve(y_test, y_proba_test, best_model_name)
        plt.savefig(output_dir / 'best_model_pr_curve.png')
        plt.close()
    
    # Save final results
    pd.DataFrame([test_metrics]).to_csv(output_dir / 'best_model_test_metrics.csv', index=False)
    
    logger.info("Heart disease prediction model improvement completed")

if __name__ == "__main__":
    main()