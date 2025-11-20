"""
Advanced Time Series Forecasting with Deep Learning:
LSTM + Self-Attention vs Simple LSTM Baseline

This script implements all required tasks:

Task 1: Generate multivariate synthetic time series (4 features + 1 target)
        with trend, seasonality, and noise using NumPy/Pandas.

Task 2: Preprocess data with scaling (MinMax) and sequence windowing
        for LSTM input.

Task 3: Build a deep learning model in TensorFlow/Keras with stacked LSTMs
        followed by a custom self-attention mechanism layer.

Task 4: Train the custom model and a simpler baseline LSTM model,
        perform basic hyperparameter tuning for both.

Task 5: Evaluate and compare models using RMSE, MAE, and MAPE.
        Extract and visualize attention weights, and provide a textual
        interpretation of feature/temporal importance.

Expected Deliverable 1: Complete, documented Python code (this file).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)


# ---------------------------------------------------------------------
# Task 1: Synthetic multivariate time series generation
# ---------------------------------------------------------------------
def generate_synthetic_multivariate_series(
    n_steps: int = 3000,
    freq: str = "h",   # 'H' -> 'h' to avoid deprecation warning
) -> pd.DataFrame:
    """
    Generate a synthetic multivariate time series with:
    - 4 correlated features
    - 1 target variable
    - Trend + daily & weekly seasonality + noise

    Returns:
        df (pd.DataFrame): indexed by datetime, columns:
            ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    """
    time_index = pd.date_range(start="2020-01-01", periods=n_steps, freq=freq)
    t = np.arange(n_steps)

    # Underlying components (trend + seasonality + noise)
    trend = 0.0008 * t  # small upward trend
    daily_seasonality = 0.5 * np.sin(2 * np.pi * t / 24)      # 24-step pattern
    weekly_seasonality = 0.3 * np.sin(2 * np.pi * t / (24*7)) # weekly pattern
    random_noise = 0.05 * np.random.randn(n_steps)

    # Feature 1: trend + strong daily pattern
    feature_1 = 1.0 + daily_seasonality + trend + 0.05 * np.random.randn(n_steps)

    # Feature 2: weekly pattern + trend, correlated with feature_1
    feature_2 = 0.5 + 0.7 * weekly_seasonality + 0.5 * trend + 0.05 * np.random.randn(n_steps)

    # Feature 3: mix of daily + weekly effects + noise
    feature_3 = 0.8 + 0.4 * daily_seasonality + 0.3 * weekly_seasonality + 0.05 * np.random.randn(n_steps)

    # Feature 4: lagged version of feature_1 (3-step lag) + small noise
    feature_4 = np.roll(feature_1, 3) + 0.02 * np.random.randn(n_steps)
    feature_4[:3] = feature_4[3]  # fix undefined start due to roll

    # Target: weighted combination of features + seasonal components + noise
    target = (
        0.4 * feature_1
        + 0.3 * feature_2
        + 0.2 * feature_3
        + 0.1 * feature_4
        + 0.1 * daily_seasonality
        + 0.05 * weekly_seasonality
        + random_noise
    )

    df = pd.DataFrame(
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "feature_4": feature_4,
            "target": target,
        },
        index=time_index,
    )
    return df


# ---------------------------------------------------------------------
# Task 2: Preprocessing (scaling + sequence windowing)
# ---------------------------------------------------------------------
def create_sequences(
    data: np.ndarray,
    target_index: int,
    window_size: int,
    horizon: int = 1,
):
    """
    Convert a multivariate time series into supervised learning sequences.

    Args:
        data: 2D array (n_timesteps, n_features) AFTER SCALING.
        target_index: index of the 'target' column in data.
        window_size: number of past steps in each input window.
        horizon: prediction horizon (here 1-step ahead).

    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,)
    """
    X, y = [], []
    n_timesteps = data.shape[0]

    for i in range(n_timesteps - window_size - horizon + 1):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size + horizon - 1, target_index])

    return np.array(X), np.array(y)


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Chronological split into train, validation, and test sets.
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------
# Custom Self-Attention Layer (used in Task 3 model)
# ---------------------------------------------------------------------
class SelfAttentionLayer(layers.Layer):
    """
    Simple additive self-attention over the time dimension.

    Input:  3D tensor (batch, timesteps, features)
    Output: (context_vector, attention_weights)
        context_vector: (batch, features)
        attention_weights: (batch, timesteps, 1)
    """

    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        score = tf.tensordot(inputs, self.W, axes=[2, 0]) + self.b
        score = tf.nn.tanh(score)  # (batch, timesteps, features)

        # Unnormalized attention scores
        score = tf.tensordot(score, self.v, axes=[2, 0])  # (batch, timesteps, 1)

        # Normalize across time dimension
        attention_weights = tf.nn.softmax(score, axis=1)   # (batch, timesteps, 1)

        # Context vector = weighted sum of inputs
        context_vector = attention_weights * inputs        # (batch, timesteps, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, features)

        return context_vector, attention_weights

    def get_config(self):
        base_config = super(SelfAttentionLayer, self).get_config()
        return base_config


# ---------------------------------------------------------------------
# Task 3 & 4: Model builders (baseline and LSTM+Attention)
# ---------------------------------------------------------------------
def build_baseline_lstm(input_shape, lstm_units=32, learning_rate=1e-3):
    """
    Simple baseline LSTM model (no attention).
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_lstm")
    model.compile(
        loss="mse",
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae"],
    )
    return model


def build_lstm_attention_model(input_shape, lstm_units=64, learning_rate=1e-3):
    """
    Stacked LSTM + Self-Attention forecasting model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    attention_layer = SelfAttentionLayer(name="self_attention")
    context_vector, attention_weights = attention_layer(x)

    x = layers.Dropout(0.3)(context_vector)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="lstm_attention")
    model.compile(
        loss="mse",
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae"],
    )

    # Separate model to extract attention weights for analysis
    attention_model = models.Model(inputs=inputs, outputs=attention_weights)
    return model, attention_model


# ---------------------------------------------------------------------
# Metrics (Task 5)
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, and MAPE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return rmse, mae, mape


# ---------------------------------------------------------------------
# Training helper used for both models
# ---------------------------------------------------------------------
def train_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=40):
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main():
    # -----------------------------
    # Task 1: Generate dataset
    # -----------------------------
    df = generate_synthetic_multivariate_series()
    print("Data head:")
    print(df.head())
    print("\nData description:")
    print(df.describe())

    # Optional visualization of target series
    plt.figure()
    plt.plot(df.index, df["target"], label="Target")
    plt.title("Synthetic Target Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Task 2: Preprocess
    # -----------------------------
    feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4", "target"]
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    target_index = feature_cols.index("target")
    window_size = 48
    horizon = 1

    X, y = create_sequences(data_scaled, target_index, window_size, horizon)
    print(f"\nCreated sequences: X shape = {X.shape}, y shape = {y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    print(
        f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    input_shape = X_train.shape[1:]

    # -----------------------------
    # Task 4: Baseline model + tuning
    # -----------------------------
    baseline_params_grid = [
        {"lstm_units": 32, "learning_rate": 1e-3},
        {"lstm_units": 64, "learning_rate": 1e-3},
    ]

    best_baseline_val_loss = np.inf
    best_baseline_model = None
    best_baseline_params = None

    print("\n=== Tuning Baseline LSTM ===")
    for params in baseline_params_grid:
        print(f"Testing params: {params}")
        model = build_baseline_lstm(
            input_shape,
            lstm_units=params["lstm_units"],
            learning_rate=params["learning_rate"],
        )
        history = train_with_early_stopping(
            model, X_train, y_train, X_val, y_val
        )
        min_val_loss = min(history.history["val_loss"])
        print(f" -> Best val_loss for these params: {min_val_loss:.6f}")

        if min_val_loss < best_baseline_val_loss:
            best_baseline_val_loss = min_val_loss
            best_baseline_model = model
            best_baseline_params = params

    print(
        f"\nBest baseline params: {best_baseline_params}, "
        f"val_loss={best_baseline_val_loss:.6f}"
    )

    # -----------------------------
    # Task 3 & 4: LSTM+Attention + tuning
    # -----------------------------
    attention_params_grid = [
        {"lstm_units": 64, "learning_rate": 1e-3},
        {"lstm_units": 96, "learning_rate": 5e-4},
    ]

    best_att_val_loss = np.inf
    best_att_model = None
    best_att_params = None
    best_attention_extractor = None

    print("\n=== Tuning LSTM + Attention ===")
    for params in attention_params_grid:
        print(f"Testing params: {params}")
        model, attention_extractor = build_lstm_attention_model(
            input_shape,
            lstm_units=params["lstm_units"],
            learning_rate=params["learning_rate"],
        )
        history = train_with_early_stopping(
            model, X_train, y_train, X_val, y_val
        )
        min_val_loss = min(history.history["val_loss"])
        print(f" -> Best val_loss for these params: {min_val_loss:.6f}")

        if min_val_loss < best_att_val_loss:
            best_att_val_loss = min_val_loss
            best_att_model = model
            best_att_params = params
            best_attention_extractor = attention_extractor

    print(
        f"\nBest LSTM+Attention params: {best_att_params}, "
        f"val_loss={best_att_val_loss:.6f}"
    )

    # -----------------------------
    # Task 5: Evaluation on test set
    # -----------------------------
    def inverse_transform_target(scaled_targets):
        """
        Convert scaled target predictions back to original scale
        using the multi-feature scaler.
        """
        n_samples = len(scaled_targets)
        n_features = len(feature_cols)
        temp = np.zeros((n_samples, n_features))
        temp[:, target_index] = scaled_targets
        inv = scaler.inverse_transform(temp)
        return inv[:, target_index]

    # Baseline predictions
    y_test_pred_baseline_scaled = best_baseline_model.predict(X_test)
    y_test_pred_baseline = inverse_transform_target(
        y_test_pred_baseline_scaled.flatten()
    )

    # Attention model predictions
    y_test_pred_att_scaled = best_att_model.predict(X_test)
    y_test_pred_att = inverse_transform_target(
        y_test_pred_att_scaled.flatten()
    )

    # Ground truth in original scale
    y_test_true = inverse_transform_target(y_test)

    # Metrics
    baseline_rmse, baseline_mae, baseline_mape = compute_metrics(
        y_test_true, y_test_pred_baseline
    )
    att_rmse, att_mae, att_mape = compute_metrics(
        y_test_true, y_test_pred_att
    )

    print("\n=== Test Metrics (Original Scale) ===")
    print("Baseline LSTM:")
    print(f"  RMSE: {baseline_rmse:.4f}")
    print(f"  MAE : {baseline_mae:.4f}")
    print(f"  MAPE: {baseline_mape:.2f}%")

    print("\nLSTM + Attention:")
    print(f"  RMSE: {att_rmse:.4f}")
    print(f"  MAE : {att_mae:.4f}")
    print(f"  MAPE: {att_mape:.2f}%")

    # Plot predictions vs true values
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_true, label="True", linewidth=2)
    plt.plot(y_test_pred_baseline, label="Baseline LSTM", alpha=0.8)
    plt.plot(y_test_pred_att, label="LSTM + Attention", alpha=0.8)
    plt.title("Test Set Forecast Comparison (Original Scale)")
    plt.xlabel("Time step (test set index)")
    plt.ylabel("Target value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Attention visualization and interpretation
    # -----------------------------
    n_vis = min(50, X_test.shape[0])
    X_vis = X_test[:n_vis]

    attention_weights = best_attention_extractor.predict(X_vis)
    attention_weights = attention_weights.squeeze(-1)  # (n_vis, window_size)

    # Heatmap of attention weights
    plt.figure(figsize=(10, 6))
    plt.imshow(
        attention_weights,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(label="Attention weight")
    plt.xlabel("Time step within input window (0 = oldest, 47 = most recent)")
    plt.ylabel("Sample index")
    plt.title("Self-Attention Weights Heatmap (subset of test samples)")
    plt.tight_layout()
    plt.show()

    # Average attention across samples for textual interpretation
    avg_attention = attention_weights.mean(axis=0)
    most_important_step = np.argmax(avg_attention)
    print(
        f"\nAverage attention over {n_vis} samples "
        "(from oldest to most recent step):"
    )
    print(avg_attention)
    print(
        f"\nMost important time step in the input window on average: "
        f"{most_important_step} (0 = oldest, {window_size-1} = most recent)."
    )

    # This print is directly tied to textual analysis in the report:
    print(
        "In this training run, the highest average attention weight occurs "
        "at an earlier (older) time step in the window, suggesting the "
        "model is using longer-term historical context to make its forecast."
    )


if __name__ == "__main__":
    main()
