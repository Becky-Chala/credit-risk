# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


def load_data(path='data/processed/data_with_target.csv'):
    """Load preprocessed data."""
    return pd.read_csv(path)
def split_data(data):
    """Split data into train and test sets, drop non-numeric features."""
    X = data.drop(columns=['is_high_risk', 'CustomerId', 'TransactionId', 'TransactionStartTime'], errors='ignore')
    # Keep only numeric columns
    X = X.select_dtypes(include=['number'])
    y = data['is_high_risk']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }


def log_model_to_mlflow(model, metrics, params, model_name):
    """Log model and metrics to MLflow."""
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.end_run()


def main():
    # Load and split data
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    # Train Logistic Regression
    print("Training Logistic Regression...")
    logreg = train_logistic_regression(X_train, y_train)
    logreg_metrics = evaluate_model(logreg, X_test, y_test)
    print("Logistic Regression Metrics:", logreg_metrics)
    log_model_to_mlflow(logreg, logreg_metrics, logreg.get_params(), "logistic_regression")

    # Train Random Forest
    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    print("Random Forest Metrics:", rf_metrics)
    log_model_to_mlflow(rf, rf_metrics, rf.get_params(), "random_forest")


if __name__ == "__main__":
    main()
