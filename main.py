#!/usr/bin/env python
"""
Main script for the heart disease prediction project.
This script runs the full pipeline from data loading to model evaluation.
"""

import os
import logging
import argparse
import time # To time operations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold # For tuning
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# Import project modules
from src.config import (
    ROOT_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, CV_FOLDS,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, SMOTE_PARAMS,
    CV_FOLDS, XGBOOST_PARAMS, VOTING_WEIGHTS, SCORING, FAIRNESS_SLICES,
    AGE_BANDS, F_BETA_SCORE_BETA, DEFAULT_MIN_PRECISION, LOGISTIC_REGRESSION_GRID,
    RANDOM_FOREST_GRID, XGBOOST_GRID, THRESHOLD_MIN_PRECISION_CONSTRAINTS
)
from src.data.data_loader import load_data, split_features_target, save_processed_data
from src.data.preprocessor import (
    check_data_quality, clean_data, create_preprocessing_pipeline, encode_target
)
from src.features.feature_engineering import (
    create_interaction_features, compute_lipid_ratio, get_feature_names,
    create_age_bands, calculate_framingham_score, calculate_ascvd_score, add_polynomial_features
)
from src.models.baseline import MajorityClassifier
from src.models.logistic import create_logistic_model, analyze_coefficients
from src.models.random_forest import create_random_forest_model, get_feature_importance, analyze_tree_paths
from src.models.xgboost import create_xgboost_model, get_xgboost_feature_importance
from src.models.ensemble import create_voting_classifier # Add Stacking if desired
from src.utils.fairness import compute_slice_metrics, export_fairness_report
from src.utils.metrics import cross_validate_model, find_optimal_threshold, fbeta_score, make_scorer, compare_models, evaluate_classifier
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
    parser.add_argument('--tune_models', action='store_true',
                        help='Perform hyperparameter tuning for RF and XGBoost')
    parser.add_argument('--run_cv', action='store_true', default=True, # Run CV by default
                        help='Run stratified cross-validation for model evaluation')
    parser.add_argument('--run_fairness', action='store_true',
                        help='Run fairness analysis across demographic slices')
    
    return parser.parse_args()

def main():
    """Run the heart disease prediction pipeline."""
    args = parse_arguments()
    start_time = time.time()
    
    # Set up paths
    data_path = args.data_path or RAW_DATA_PATH
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting ENHANCED heart disease prediction pipeline")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # --- Step 1: Load and explore data (keep existing) ---
    logger.info("STEP 1: Data loading and exploration")
    data = load_data(data_path)
    quality_summary = check_data_quality(data)

    # --- Step 2: Data preprocessing (keep existing clean_data) ---
    logger.info("STEP 2: Data preprocessing")
    # Using advanced imputation by default, can be made configurable
    cleaned_data = clean_data(data, method='advanced')

    # --- Step 3: Feature engineering (ENHANCED) ---
    logger.info("STEP 3: Feature engineering")
    # Apply new feature engineering steps [cite: 9, 11, 13]
    enhanced_data = create_age_bands(cleaned_data) # Add age bands [cite: 9]
    enhanced_data = create_interaction_features(enhanced_data) # Keep existing interactions
    enhanced_data = compute_lipid_ratio(enhanced_data) # Keep existing lipid feature
    enhanced_data = calculate_framingham_score(enhanced_data) # Add Framingham (placeholder) [cite: 11]
    enhanced_data = calculate_ascvd_score(enhanced_data) # Add ASCVD (placeholder) [cite: 11]

    # Add polynomial features BEFORE splitting X/y if they should be part of the base features
    # Identify numerical features available *after* previous steps
    current_numerical_features = enhanced_data.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COLUMN in current_numerical_features:
         current_numerical_features.remove(TARGET_COLUMN)
    # Select a subset for polynomial features if needed (e.g., top K important or clinical ones)
    poly_feature_candidates = [f for f in NUMERICAL_FEATURES if f in current_numerical_features] # Use original list or adapt
    enhanced_data = add_polynomial_features(enhanced_data, poly_feature_candidates, degree=2) # Add Poly feats [cite: 13]


    # Update feature lists based on added features (crucial for preprocessing pipeline)
    # Get all columns except target
    all_cols = enhanced_data.columns.tolist()
    if TARGET_COLUMN in all_cols:
        all_cols.remove(TARGET_COLUMN)

    # Re-identify numerical and categorical features
    final_numerical_features = enhanced_data[all_cols].select_dtypes(include=np.number).columns.tolist()
    final_categorical_features = enhanced_data[all_cols].select_dtypes(exclude=np.number).columns.tolist()
    # Add newly created categorical features like 'Age_Band' if they exist
    if 'Age_Band' in enhanced_data.columns and 'Age_Band' not in final_categorical_features:
        final_categorical_features.append('Age_Band')
    # Remove features that might have been used solely for intermediate calculations if necessary

    logger.info(f"Final numerical features ({len(final_numerical_features)}): {final_numerical_features}")
    logger.info(f"Final categorical features ({len(final_categorical_features)}): {final_categorical_features}")

    # Split features and target
    X, y = split_features_target(enhanced_data, TARGET_COLUMN)

    # Create preprocessing pipeline using FINAL feature lists
    preprocessor = create_preprocessing_pipeline(final_numerical_features, final_categorical_features)

    # --- Step 4: Train-test split (keep existing) ---
    logger.info("STEP 4: Train-test split")
    # Note: CV is now the primary evaluation method, test set is for final validation/fairness
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Step 5: Handle Class Imbalance (keep existing SMOTE option, but CV handles weights better) ---
    # SMOTE is generally applied *only* to training data, *within* each CV fold or before final training.
    # Applying it before splitting or CV introduces data leakage.
    # Class weights within models are often preferred with CV.
    if args.use_smote:
         logger.warning("Applying SMOTE before CV is generally discouraged due to data leakage risk. "
                        "Consider using class_weight='balanced' or SMOTE within a CV pipeline.")
         # If proceeding, apply only to training set AFTER split
         # X_train_processed_for_smote = preprocessor.fit_transform(X_train) # Fit on train
         # X_train_resampled, y_train_resampled = apply_smote(X_train_processed_for_smote, y_train, **SMOTE_PARAMS)
         # Need to handle preprocessor fitting carefully if SMOTE is applied outside pipeline
         # For simplicity, we'll rely on class_weight='balanced' which is set in config

    # --- Step 6: Define Models ---
    logger.info("STEP 6: Defining models")
    # We will use pipelines for preprocessing + model

    # Define base model pipelines (ensure class_weight='balanced' is set in config params)
    log_pipe = create_logistic_model(preprocessor, **LOGISTIC_REGRESSION_PARAMS)
    rf_pipe = create_random_forest_model(preprocessor, **RANDOM_FOREST_PARAMS)
    xgb_pipe, xgb_fit_params = create_xgboost_model(preprocessor, **XGBOOST_PARAMS) # xgb_fit_params might contain early stopping

    # --- Step 7: Optional Hyperparameter Tuning ---
    if args.tune_models:
        logger.info("STEP 7: Hyperparameter Tuning (Optional)")
        # tuning_scorer = make_scorer(recall_score, zero_division=0) # Tune based on recall
        tuning_scorer = make_scorer(recall_score, zero_division=0)

        # Tune Logistic Regression
        logger.info("Tuning Logistic Regression...")
        log_grid_search = GridSearchCV(log_pipe, LOGISTIC_REGRESSION_GRID, cv=StratifiedKFold(CV_FOLDS), scoring=tuning_scorer, n_jobs=-1, verbose=1)
        log_grid_search.fit(X_train, y_train)
        logger.info(f"Best Log Params: {log_grid_search.best_params_}")
        logger.info(f"Best Log Recall Score: {log_grid_search.best_score_:.4f}")
        log_pipe = log_grid_search.best_estimator_ # Use best pipeline

        # Tune Random Forest
        logger.info("Tuning Random Forest...")
        rf_grid_search = GridSearchCV(rf_pipe, RANDOM_FOREST_GRID, cv=StratifiedKFold(CV_FOLDS), scoring=tuning_scorer, n_jobs=-1, verbose=1)
        rf_grid_search.fit(X_train, y_train)
        logger.info(f"Best RF Params: {rf_grid_search.best_params_}")
        logger.info(f"Best RF Recall Score: {rf_grid_search.best_score_:.4f}")
        rf_pipe = rf_grid_search.best_estimator_ # Use best pipeline

        # Tune XGBoost
        logger.info("Tuning XGBoost...")
        xgb_grid_search = GridSearchCV(xgb_pipe, XGBOOST_GRID, cv=StratifiedKFold(CV_FOLDS), scoring=tuning_scorer, n_jobs=-1, verbose=1)
        # Need to handle fit_params for early stopping if tuning it
        xgb_grid_search.fit(X_train, y_train) # Add fit_params=xgb_fit_params if needed and compatible with GridSearchCV
        logger.info(f"Best XGBoost Params: {xgb_grid_search.best_params_}")
        logger.info(f"Best XGBoost Recall Score: {xgb_grid_search.best_score_:.4f}")
        xgb_pipe = xgb_grid_search.best_estimator_ # Use best pipeline

    # --- Step 8: Cross-Validation Evaluation ---
    cv_results = {}
    if args.run_cv:
        logger.info(f"STEP 8: Running {CV_FOLDS}-Fold Cross-Validation")
        models_to_cv = {
            "Logistic Regression": log_pipe,
            "Random Forest": rf_pipe,
            "XGBoost": xgb_pipe
        }
        for name, model in models_to_cv.items():
            logger.info(f"--- Cross-validating {name} ---")
            # Use appropriate fit_params for XGBoost if needed
            fit_params = xgb_fit_params if name == "XGBoost" else None
            cv_results[name] = cross_validate_model(model, X_train, y_train, cv_folds=CV_FOLDS, scoring=SCORING, fit_params=fit_params)

        # Compare models based on CV results
        cv_comparison_df = compare_models(cv_results, target_metric='recall') # Focus on recall
        logger.info("\nCross-Validation Comparison Table:\n" + cv_comparison_df.to_string())
        cv_comparison_df.to_csv(output_dir / 'cv_model_comparison.csv', index=False)

    # --- Step 9: Train Final Models & Evaluate on Test Set ---
    logger.info("STEP 9: Training final models on full training data and evaluating on test set")
    final_models = {}
    test_predictions = {}
    test_probabilities = {}
    test_metrics = {}
    optimal_thresholds = {}

    # Fit final models (LR, RF, XGB)
    logger.info("Fitting Logistic Regression...")
    # Ensure pipeline is fitted correctly
    log_pipe.fit(X_train, y_train)
    final_models['Logistic Regression'] = log_pipe

    logger.info("Fitting Random Forest...")
    # Ensure pipeline is fitted correctly
    rf_pipe.fit(X_train, y_train)
    final_models['Random Forest'] = rf_pipe

    logger.info("Fitting XGBoost...")
    # Ensure pipeline is fitted correctly
    xgb_pipe.fit(X_train, y_train) # Add fit_params=fit_params_final if using early stopping
    final_models['XGBoost'] = xgb_pipe

    # Fit Baseline Model
    logger.info("Fitting Baseline (Majority Classifier)...")
    baseline = MajorityClassifier()
    # Baseline fits directly on y_train (doesn't use X_train features)
    baseline.fit(X_train, y_train) # Pass X_train for API compatibility, but it's ignored
    final_models['Baseline'] = baseline

    # Add Ensemble Model (Voting)
    logger.info("Creating and fitting Voting Ensemble...")
    # Ensure base estimators are the *fitted* pipelines/models
    estimators = [
        ('lr', final_models['Logistic Regression']),
        ('rf', final_models['Random Forest']),
        ('xgb', final_models['XGBoost'])
    ]
    voting_clf = create_voting_classifier(estimators, voting='soft', weights=list(VOTING_WEIGHTS.values()) if VOTING_WEIGHTS else None)
    # Fit the ensemble itself (uses predict/predict_proba of fitted base estimators)
    # Technically, fitting VotingClassifier refits if base estimators aren't fitted,
    # but fitting them explicitly above is clearer.
    # Re-fitting here shouldn't be necessary if base models are already fitted.
    # Let's fit it just to be safe with the API.
    voting_clf.fit(X_train, y_train) # This fit might be redundant depending on VotingClassifier version/usage
    final_models['Voting Ensemble'] = voting_clf

    # Evaluate on Test Set
    logger.info("\n--- Test Set Evaluation (Before Threshold Optimization) ---")
    test_results_raw = {}
    # Make sure to evaluate all models, including Baseline
    for name, model in final_models.items():
        logger.info(f"Evaluating {name} on test set...")
        y_pred_test = model.predict(X_test)
        # Handle probability prediction for different model types
        if hasattr(model, "predict_proba"):
             y_proba_test = model.predict_proba(X_test)[:, 1]
        else:
             y_proba_test = None # Baseline won't have probabilities
             if name != 'Baseline': # Avoid warning for baseline
                 logger.warning(f"Model {name} does not support predict_proba. ROC/PR AUC not available.")

        test_predictions[name] = y_pred_test
        test_probabilities[name] = y_proba_test
        test_results_raw[name] = evaluate_classifier(y_test, y_pred_test, y_proba_test)

    # Compare raw test results (including Baseline)
    test_comparison_raw_df = compare_models(test_results_raw, target_metric='recall')
    logger.info("\nTest Set Comparison Table (Default Threshold 0.5):\n" + test_comparison_raw_df.to_string())
    test_comparison_raw_df.to_csv(output_dir / 'test_model_comparison_raw.csv', index=False)

    # --- Step 10: Threshold Optimization on Test Set ---
    logger.info("\nSTEP 10: Optimizing Thresholds on Test Set Results")
    test_results_optimized = {}
    # Iterate through all models evaluated previously
    for name in test_results_raw.keys():
         model = final_models[name] # Get the fitted model instance
         y_proba_test = test_probabilities.get(name)
         if y_proba_test is not None: # Can only optimize threshold if probas exist
             logger.info(f"Optimizing threshold for {name}...")
             opt_threshold, opt_metrics = find_optimal_threshold(
                 y_test, y_proba_test, optimize_for='recall',
                 min_precision_constraints=THRESHOLD_MIN_PRECISION_CONSTRAINTS
             )
             optimal_thresholds[name] = opt_threshold
             test_results_optimized[name] = opt_metrics
             logger.info(f"Optimized Threshold for {name}: {opt_threshold:.4f}")
             # Update predictions using the optimal threshold
             test_predictions[name] = (y_proba_test >= opt_threshold).astype(int)
         else:
             logger.info(f"Cannot optimize threshold for {name} (no probabilities). Using raw results.")
             # Use the raw results directly for models without probabilities (like Baseline)
             test_results_optimized[name] = test_results_raw[name]
             optimal_thresholds[name] = 0.5 # Assign default threshold

    # Compare optimized test results (includes Baseline evaluated at default 0.5)
    test_comparison_opt_df = compare_models(test_results_optimized, target_metric='recall')
    logger.info("\nTest Set Comparison Table (Optimized Thresholds):\n" + test_comparison_opt_df.to_string())
    test_comparison_opt_df.to_csv(output_dir / 'test_model_comparison_optimized.csv', index=False)


    # --- Step 11: Fairness Analysis (Optional) ---
    if args.run_fairness:
        logger.info("\nSTEP 11: Fairness Analysis")
        # Choose the best model based on optimized recall on test set
        best_model_name = test_comparison_opt_df.iloc[0]['Model']
        logger.info(f"Running fairness analysis for best model: {best_model_name}")

        y_pred_best = test_predictions[best_model_name]
        y_proba_best = test_probabilities[best_model_name]

        # Ensure X_test includes the slice features (Age_Band was added earlier)
        # Make sure other slice features like 'Gender', 'Diabetes' are in X_test
        slice_features_present = [f for f in FAIRNESS_SLICES if f in X_test.columns]
        if not slice_features_present:
             logger.warning("No specified slice features found in X_test columns. Skipping fairness analysis.")
        else:
             fairness_metrics_df = compute_slice_metrics(X_test, y_test, y_pred_best, y_proba_best, slice_features_present)
             if not fairness_metrics_df.empty:
                 logger.info("\nFairness Metrics Table:\n" + fairness_metrics_df.to_string())
                 export_fairness_report(fairness_metrics_df, output_dir / 'fairness_report.csv', format='csv')
                 export_fairness_report(fairness_metrics_df, output_dir / 'fairness_report.md', format='markdown')


    # --- Step 12: Final Reporting & Saving Artifacts ---
    logger.info("\nSTEP 12: Final Reporting and Saving Artifacts")

    # Plot overall model comparison based on OPTIMIZED test results
    logger.info("Plotting overall model comparison...")
    plot_model_comparison(test_comparison_opt_df) # Plot comparison of optimized results
    plt.title("Model Comparison on Test Set (Optimized Thresholds)")
    plt.savefig(output_dir / 'model_comparison_optimized.png')
    plt.close()

    logger.info("Generating individual model plots (Confusion Matrix, ROC, PR)...")
    # Generate plots for ALL models using OPTIMIZED predictions/thresholds
    for name, model in final_models.items():
        logger.info(f"--- Generating plots for: {name} ---")
        y_pred_opt = test_predictions[name] # Use predictions after threshold optimization
        y_proba_opt = test_probabilities[name] # Use original probabilities

        # Plot Confusion Matrix for all models
        plot_confusion_matrix(y_test, y_pred_opt, class_names=["No Disease", "Heart Disease"])
        plot_title = f"Confusion Matrix: {name}"
        if name != 'Baseline': # Add threshold info if applicable
             plot_title += f" (Threshold={optimal_thresholds[name]:.3f})"
        plt.suptitle(plot_title)
        plt.savefig(output_dir / f'{name.replace(" ", "_").lower()}_confusion_matrix_optimized.png')
        plt.close()

        # Plot ROC and PR curves only if probabilities exist
        if y_proba_opt is not None:
            # Plot ROC curve
            plot_roc_curve(y_test, y_proba_opt, name)
            plt.savefig(output_dir / f'{name.replace(" ", "_").lower()}_roc_curve.png')
            plt.close()
            # Plot Precision-Recall curve
            plot_precision_recall_curve(y_test, y_proba_opt, name)
            plt.savefig(output_dir / f'{name.replace(" ", "_").lower()}_pr_curve.png')
            plt.close()
        else:
             logger.info(f"Skipping ROC/PR plots for {name} (no probabilities).")


    # Save feature importance for the best model (or selected models)
    # (Keep existing logic for saving feature importance of the best model)
    # You could extend this to save importance for RF, XGB if desired
    best_model_name = test_comparison_opt_df.iloc[0]['Model']
    logger.info(f"Saving feature importance for best model: {best_model_name}")
    try:
        best_model_instance = final_models[best_model_name]
        if hasattr(best_model_instance, 'named_steps') and 'preprocessor' in best_model_instance.named_steps:
             fitted_preprocessor = best_model_instance.named_steps['preprocessor']
             # Ensure preprocessor is fitted before getting feature names
             # It should be fitted from the pipeline fitting earlier
             try:
                  feature_names = get_feature_names(fitted_preprocessor)
             except Exception as e_feat:
                  logger.error(f"Could not get feature names from preprocessor for {best_model_name}: {e_feat}")
                  feature_names = None # Fallback

             if feature_names:
                 importance_df = None
                 # Add logic to get importance based on best_model_name as before...
                 if 'Logistic Regression' in best_model_name:
                     importance_df = analyze_coefficients(best_model_instance, feature_names)
                 elif 'Random Forest' in best_model_name:
                      gini_importances = best_model_instance.named_steps['classifier'].feature_importances_
                      importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': gini_importances}).sort_values('Importance', ascending=False)
                 elif 'XGBoost' in best_model_name:
                     importance_df = get_xgboost_feature_importance(best_model_instance, feature_names)
                 # Add logic for Voting/Stacking if needed

                 if importance_df is not None:
                     plot_feature_importance(importance_df.head(20), f"{best_model_name} - Feature Importance")
                     plt.savefig(output_dir / f'{best_model_name.replace(" ", "_").lower()}_feature_importance.png')
                     plt.close()
                     importance_df.to_csv(output_dir / f'{best_model_name.replace(" ", "_").lower()}_feature_importance.csv', index=False)
                     logger.info(f"Feature importance saved for {best_model_name}.")
        else:
             logger.warning(f"Cannot extract feature importance for model type: {best_model_name} (not a pipeline or no preprocessor step).")

    except Exception as e:
        logger.error(f"Error saving feature importance for {best_model_name}: {e}")

    end_time = time.time()
    logger.info(f"Pipeline finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
