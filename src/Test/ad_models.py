import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_if(
    X_train: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42
):
    """
    Train Isolation Forest using ONLY normal samples.
    Returns a sklearn Pipeline: StandardScaler -> IsolationForest
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train)
    return pipeline


def predict_if(pipeline, X: np.ndarray):
    """
    Predict anomalies using a trained IF pipeline.

    Returns:
        y_pred: 0 = normal, 1 = anomaly
        anomaly_score: higher = more anomalous
    """
    pred = pipeline.predict(X)          # 1 normal, -1 anomaly
    y_pred = (pred == -1).astype(int)

    # continuous anomaly score
    anomaly_score = -pipeline.decision_function(X)

    return y_pred, anomaly_score


def save_model(pipeline, path: str):
    joblib.dump(pipeline, path)


def load_model(path: str):
    return joblib.load(path)
