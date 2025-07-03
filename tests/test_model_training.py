import pandas as pd
from src.model_training import split_data, evaluate_model
from sklearn.linear_model import LogisticRegression


def test_split_data_shapes():
    """Test if split_data returns correct shapes."""
    dummy_data = pd.DataFrame({
        'feature1': range(20),
        'feature2': range(20, 40),
        'is_high_risk': [0, 1] * 10  # 10 of each class
    })
    X_train, X_test, y_train, y_test = split_data(dummy_data)
    assert len(X_train) + len(X_test) == len(dummy_data)
    assert len(y_train) + len(y_test) == len(dummy_data)
def test_evaluate_model_keys():
    """Test if evaluate_model returns all required metrics."""
    X_train = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
    y_train = pd.Series([0, 1])
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        assert metric in metrics
