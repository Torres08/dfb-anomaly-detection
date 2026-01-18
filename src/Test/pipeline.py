import os
import numpy as np
from sklearn.metrics import classification_report

from ad_models import train_if, predict_if, save_model


def main():
    rng = np.random.RandomState(42)

    # -------------------------------------------------
    # 1) Dummy latent vectors
    # -------------------------------------------------

    # Train: ONLY normal samples
    X_train = rng.normal(loc=0.0, scale=1.0, size=(1000, 64))

    # Test: normal + anomalies
    X_test_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 64))
    X_test_anomaly = rng.normal(loc=4.0, scale=1.0, size=(50, 64))

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * 200 + [1] * 50)  # 0 normal, 1 anomaly

    # -------------------------------------------------
    # 2) Train Isolation Forest
    # -------------------------------------------------

    pipeline = train_if(
        X_train,
        contamination=0.2  # 50 / 250 ≈ 0.2
    )

    # -------------------------------------------------
    # 3) Predict and evaluate
    # -------------------------------------------------

    y_pred, anomaly_score = predict_if(pipeline, X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------------------------------------
    # 4) Save model
    # -------------------------------------------------

    os.makedirs("models", exist_ok=True)
    save_model(pipeline, "models/if_dummy.pkl")
    print("Model saved to models/if_dummy.pkl")


if __name__ == "__main__":
    main()
