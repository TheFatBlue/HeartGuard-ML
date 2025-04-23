# HeartGuard-ML

This project aims to predict heart disease status based on various patient health indicators, helping to identify individuals at risk and recommend early intervention or further diagnostic tests.

## Project Overview

We implement multiple machine learning models to predict whether a patient has heart disease based on their health and lifestyle features. The project focuses on maximizing recall (sensitivity) to minimize the risk of false negatives, as missing a positive case of heart disease could lead to serious health consequences.

## Dataset

The dataset contains 10,000 individual records with various health-related indicators and risk factors linked to heart disease, including:

- Demographic information (Age, Gender)
- Clinical measurements (Blood Pressure, Cholesterol Level, BMI, etc.)
- Lifestyle factors (Exercise Habits, Smoking, Alcohol Consumption, etc.)
- Medical history (Family Heart Disease, Diabetes, etc.)

The target variable is `Heart Disease Status` (Yes/No) with an imbalanced class distribution (20% positive, 80% negative).

## Project Structure

```
heart_disease_prediction/
|
├── main.py
├── notebooks
│   └── 01_data_exploration.ipynb
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── config.py
    ├── data
    │   ├── __init__.py
    │   ├── data_loader.py
    │   ├── download.py
    │   ├── imputation.py
    │   └── preprocessor.py
    ├── features
    │   ├── __init__.py
    │   ├── advanced_features.py
    │   └── feature_engineering.py
    ├── models
    │   ├── __init__.py
    │   ├── baseline.py
    │   ├── ensemble.py
    │   ├── hyperparameter_tuning.py
    │   ├── logistic.py
    │   ├── random_forest.py
    │   └── xgboost.py
    ├── utils
    │   ├── __init__.py
    │   ├── fairness.py
    │   ├── imbalance.py
    │   └── metrics.py
    └── visualization
        ├── __init__.py
        └── visualize.py
```

## Installation

1. Clone this repository
2. Create a conda environment (recommended)
   ```
   conda create -n hgml python=3.11
   conda activate hgml
   ```
3. Install dependencies
   ```
   pip install uv
   uv pip install -r requirements.txt
   ```

4. Download the dataset
   ```
   python src/data/download.py
   ```

## Usage

### Running the Full Pipeline

```bash
python main.py --data_path data/raw/heart_disease.csv --use_smote --tune_models --run_fairness
```

Options:
- `--data_path`: Path to the raw data CSV (default: from config)
- `--output_dir`: Directory to save output files (default: from config)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--use_smote`: Use SMOTE for handling class imbalance
- `--skip_eda`: Skip exploratory data analysis visualizations
- `--tune_models`: Perform hyperparameter tuning for LR, RF and XGBoost
- `--run_cv`: Run stratified cross-validation for model evaluation
- `--run_fairness`: Run fairness analysis across demographic slices

### Exploring the Data

Open the Jupyter notebook for exploratory data analysis:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Evaluation Metrics

We focus on the following metrics:

- **Recall**: The ability to correctly identify positive cases (primary metric)
- **Precision**: The accuracy of positive predictions
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was created for CSC522 group project
- Based on heart disease risk factors identified by the American College of Cardiology/American Heart Association
