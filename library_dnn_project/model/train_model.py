"""
train_model.py
---------------
Trains a Deep Neural Network (Keras / TensorFlow) that predicts a book's
BORROW DEMAND for the coming week, from historical + calendar + catalog
features (see feature_engineering.py for the full feature list).

WHY A DNN (and not a plain heuristic / linear model)?
------------------------------------------------------
Demand depends on several signals that interact NON-LINEARLY:
    - recent borrowing trend (lag_1..lag_4, rolling_avg_4)
    - whether the week is inside an exam window (is_exam_period)
    - which category/department the book belongs to
      (exam pressure hits categories differently — e.g. "Data Structures"
       spikes harder near a CS exam than "Fiction" does)
    - cyclical calendar position (month/week sine-cosine encodings)
    - supply constraints (total_copies) and long-run popularity

A single linear formula cannot capture "exam_period x category" type
interactions well. A small feed-forward DNN with a couple of hidden
layers learns these interactions automatically from data, which is
the core novelty/justification for using deep learning in this project
instead of a simple moving average.

Outputs (saved into model/artifacts/):
    demand_model.keras   - trained Keras model
    scaler.pkl            - StandardScaler fitted on training features
    feature_columns.json  - ordered feature column names
    training_history.png  - loss curve plot
    metrics.json           - MAE / RMSE / R2 on held-out test data

Run:
    python model/train_model.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from feature_engineering import get_training_table, FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(input_dim: int) -> keras.Model:
    """
    Feed-forward DNN (regression head).

    Architecture: 16 -> 64 -> 32 -> 16 -> 1, with BatchNorm + Dropout for
    regularisation (dataset is tabular/moderate-sized, so a compact network
    avoids overfitting while still modelling non-linear feature interactions).
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation="relu"),

        layers.Dense(1, activation="relu"),  # demand counts are >= 0
    ], name="book_demand_dnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    print("Loading + engineering features...")
    df = get_training_table()
    print(f"  Training table: {df.shape[0]} rows, {len(FEATURE_COLUMNS)} features")

    X = df[FEATURE_COLUMNS].values.astype("float32")
    y = df[TARGET_COLUMN].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("Building DNN...")
    model = build_model(input_dim=X_train_s.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    print("Training...")
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    # -------- Evaluation --------
    y_pred = model.predict(X_test_s).flatten()
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest MAE:  {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R2:   {r2:.3f}")

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2),
               "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # -------- Save artifacts --------
    model.save(os.path.join(ARTIFACT_DIR, "demand_model.keras"))
    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # also persist the max demand observed (used later to scale the 0-100 demand score)
    with open(os.path.join(ARTIFACT_DIR, "demand_score_reference.json"), "w") as f:
        json.dump({"max_observed_weekly_demand": float(df["demand"].max()),
                    "p95_weekly_demand": float(df["demand"].quantile(0.95))}, f, indent=2)

    # -------- Loss curve plot --------
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("DNN Training History - Book Demand Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "training_history.png"), dpi=120)

    print("\nSaved artifacts to:", ARTIFACT_DIR)
    print(" - demand_model.keras")
    print(" - scaler.pkl")
    print(" - feature_columns.json")
    print(" - metrics.json")
    print(" - training_history.png")


if __name__ == "__main__":
    main()
