# src/models/ensemble.py
"""
Ensemble models (Voting, Stacking) for heart disease prediction.
"""
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

def create_voting_classifier(estimators, voting='soft', weights=None):
    """
    Create a Voting Classifier ensemble.

    Parameters:
    -----------
    estimators : list of tuple
        List of (name, estimator) tuples. Estimators should be pre-fitted
        or pipelines ready to be fitted.
    voting : str, default='soft'
        'soft' uses predicted probabilities, 'hard' uses majority rule. [cite: 8]
    weights : list or array-like, optional
        Sequence of weights for estimators. [cite: 8]

    Returns:
    --------
    VotingClassifier
    """
    logger.info(f"Creating Voting Classifier (voting='{voting}')")
    if weights:
        logger.info(f"Using weights: {weights}")

    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1 # Use all available CPUs
    )
    return ensemble

def create_stacking_classifier(estimators, final_estimator=None, cv=5):
    """
    Create a Stacking Classifier ensemble.

    Parameters:
    -----------
    estimators : list of tuple
        List of (name, estimator) tuples (base learners).
    final_estimator : estimator object, optional
        Meta-learner. Defaults to LogisticRegression. [cite: 8]
    cv : int or cross-validation generator, default=5
        Determines the cross-validation splitting strategy.

    Returns:
    --------
    StackingClassifier
    """
    logger.info("Creating Stacking Classifier")
    if final_estimator is None:
        # Default to Logistic Regression meta-learner
        # Need to ensure it handles preprocessed input from base learners
        # This might require careful pipeline construction if base learners include preprocessing
        final_estimator = LogisticRegression(class_weight='balanced', random_state=42)
        logger.info("Using default Logistic Regression as final estimator.")

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        stack_method='predict_proba', # Use probabilities for meta-learner
        n_jobs=-1
    )
    return ensemble

# Example usage within main.py might involve:
# log_pipe = create_logistic_model(...)
# rf_pipe = create_random_forest_model(...)
# xgb_pipe, _ = create_xgboost_model(...)
# estimators = [('lr', log_pipe), ('rf', rf_pipe), ('xgb', xgb_pipe)]
# voting_clf = create_voting_classifier(estimators, weights=[0.2, 0.4, 0.4])
# stacking_clf = create_stacking_classifier(estimators)
# voting_clf.fit(X_train, y_train)
# stacking_clf.fit(X_train, y_train)