"""
gradient_descent.py
--------------------
Core implementation for the Comparative Study of Batch GD, SGD, and
Mini-Batch GD, applied to a simple Linear Regression problem
(house area -> house price).

Everything here is deliberately written from scratch with plain numpy
(no sklearn) so it's easy to explain line-by-line in a viva.
"""

import time
import numpy as np


# ---------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------
def generate_house_data(n_samples=200, noise=15000.0, seed=42):
    """
    Synthetic 'house area (sq ft) vs price (INR/$, arbitrary unit)' data.
    True relationship: price = 50 * area + 50000 + noise
    """
    rng = np.random.default_rng(seed)
    area = rng.uniform(500, 4000, n_samples)          # sq ft
    price = 50 * area + 50000 + rng.normal(0, noise, n_samples)
    return area, price


def normalize(x):
    """Standardize a 1D array to mean 0, std 1. Returns (x_norm, mean, std)."""
    mean, std = x.mean(), x.std()
    std = std if std > 1e-12 else 1.0
    return (x - mean) / std, mean, std


def denormalize_params(w_norm, b_norm, x_mean, x_std, y_mean, y_std):
    """
    Convert parameters learned on normalized x and y
    back to the original units.
    """

    w_real = (y_std * w_norm) / x_std

    b_real = (
        y_mean
        + y_std * b_norm
        - w_real * x_mean
    )
    return w_real, b_real


def mse_loss(x, y, w, b):
    """Mean Squared Error for y = w*x + b."""
    y_pred = w * x + b
    return np.mean((y_pred - y) ** 2)


def predict(x, w, b):
    return w * x + b


# ---------------------------------------------------------------------
# 2. The three Gradient Descent variants
# ---------------------------------------------------------------------
# All three share the same MSE loss:  L = (1/m) * sum((w*x+b - y)^2)
# Gradients (for a batch of size m):
#   dL/dw = (2/m) * sum((w*x+b - y) * x)
#   dL/db = (2/m) * sum((w*x+b - y))
# They differ ONLY in how much data (m) is used per parameter update.

def _grad(x_batch, y_batch, w, b):
    m = len(x_batch)
    y_pred = w * x_batch + b
    error = y_pred - y_batch
    dw = (2 / m) * np.sum(error * x_batch)
    db = (2 / m) * np.sum(error)
    return dw, db


def batch_gradient_descent(x, y, lr=0.1, epochs=50, w_init=0.0, b_init=0.0):
    """
    Uses the FULL dataset for every parameter update.
    -> 1 update per epoch.
    """
    w, b = w_init, b_init
    loss_history = []
    start = time.perf_counter()

    for _epoch in range(epochs):
        dw, db = _grad(x, y, w, b)
        w -= lr * dw
        b -= lr * db
        loss_history.append(mse_loss(x, y, w, b))

    elapsed = time.perf_counter() - start
    return {
        "name": "Batch GD",
        "w": w, "b": b,
        "loss_history": loss_history,
        "n_updates": epochs,
        "time_sec": elapsed,
    }


def stochastic_gradient_descent(x, y, lr=0.01, epochs=50, w_init=0.0, b_init=0.0, seed=0):
    """
    Uses ONE randomly picked training sample per parameter update.
    -> n_samples updates per epoch.
    """
    rng = np.random.default_rng(seed)
    w, b = w_init, b_init
    loss_history = []
    n = len(x)
    start = time.perf_counter()

    for _epoch in range(epochs):
        indices = rng.permutation(n)
        for i in indices:
            xi, yi = x[i:i + 1], y[i:i + 1]
            dw, db = _grad(xi, yi, w, b)
            w -= lr * dw
            b -= lr * db
            # log loss on the FULL dataset so all 3 methods are comparable
            loss_history.append(mse_loss(x, y, w, b))

    elapsed = time.perf_counter() - start
    return {
        "name": "SGD",
        "w": w, "b": b,
        "loss_history": loss_history,
        "n_updates": epochs * n,
        "time_sec": elapsed,
    }


def mini_batch_gradient_descent(x, y, lr=0.05, epochs=50, batch_size=16,
                                 w_init=0.0, b_init=0.0, seed=0):
    """
    Uses a small BATCH of samples per parameter update.
    -> ceil(n_samples / batch_size) updates per epoch.
    """
    rng = np.random.default_rng(seed)
    w, b = w_init, b_init
    loss_history = []
    n = len(x)
    start = time.perf_counter()

    for _epoch in range(epochs):
        indices = rng.permutation(n)
        for start_idx in range(0, n, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            xb, yb = x[batch_idx], y[batch_idx]
            dw, db = _grad(xb, yb, w, b)
            w -= lr * dw
            b -= lr * db
            loss_history.append(mse_loss(x, y, w, b))

    elapsed = time.perf_counter() - start
    n_updates = epochs * int(np.ceil(n / batch_size))
    return {
        "name": "Mini-Batch GD",
        "w": w, "b": b,
        "loss_history": loss_history,
        "n_updates": n_updates,
        "time_sec": elapsed,
    }
