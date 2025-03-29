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
│
├── data/
│   ├── raw/
│   │   └── heart_disease_data.csv
│   └── processed/
│       └── heart_disease_cleaned.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Functions to load data
│   │   └── preprocessor.py    # Data preprocessing functions
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Feature transformation functions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py        # Majority classifier implementation
│   │   ├── logistic.py        # Logistic regression model
│   │   └── random_forest.py   # Random forest model
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py       # Plotting functions for EDA and model evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py         # Custom evaluation metrics
│       └── imbalance.py       # Functions to handle class imbalance
│
├── main.py                    # Script to run the full pipeline
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline

```bash
python main.py --data_path data/raw/heart_disease.csv --use_smote
```

Options:
- `--data_path`: Path to the raw data CSV (default: from config)
- `--output_dir`: Directory to save output files (default: from config)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--use_smote`: Use SMOTE for handling class imbalance
- `--skip_eda`: Skip exploratory data analysis visualizations

### Exploring the Data

Open the Jupyter notebook for exploratory data analysis:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Models

We implement and compare the following classification models:

1. **Majority Classifier** (baseline): Always predicts the most frequent class (No Heart Disease)
2. **Logistic Regression**: A linear model with interpretable coefficients
3. **Random Forest**: An ensemble tree-based model that can capture non-linear relationships

## Handling Class Imbalance

Since heart disease cases represent only 20% of the dataset, we implement several strategies to handle class imbalance:

1. **SMOTE**: Synthetic Minority Over-sampling Technique to generate synthetic samples of the minority class
2. **Class weights**: Adjusting class weights to penalize misclassifications of the minority class more heavily
3. **Threshold optimization**: Adjusting the probability threshold to maximize recall

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