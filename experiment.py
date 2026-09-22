"""
experiment.py
-------------
Runs Batch GD, SGD, and Mini-Batch GD on the SAME dataset, model,
loss function and initial parameters, then compares:
    - final loss
    - number of parameter updates
    - training time
    - convergence behaviour (loss vs iteration plot)

Run directly with:  python3 experiment.py
Produces:
    results/loss_vs_iterations.png
    results/comparison_table.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gradient_descent import (
    generate_house_data, normalize, mse_loss,
    batch_gradient_descent, stochastic_gradient_descent,
    mini_batch_gradient_descent, denormalize_params,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_all(n_samples=200, epochs=50, batch_size=16,
            lr_batch=0.05, lr_sgd=0.01, lr_minibatch=0.03, seed=42):
    """Run all three GD variants under matched conditions and return results."""
    area, price = generate_house_data(n_samples=n_samples, seed=seed)
    x_norm, x_mean, x_std = normalize(area)
    y = price  # keep target in original units; only the feature is normalized

    # Same starting point for every algorithm -> fair comparison
    w0, b0 = 0.0, 0.0

    results = {
        "Batch GD": batch_gradient_descent(x_norm, y, lr=lr_batch, epochs=epochs,
                                            w_init=w0, b_init=b0),
        "SGD": stochastic_gradient_descent(x_norm, y, lr=lr_sgd, epochs=epochs,
                                            w_init=w0, b_init=b0, seed=seed),
        "Mini-Batch GD": mini_batch_gradient_descent(x_norm, y, lr=lr_minibatch,
                                                       epochs=epochs, batch_size=batch_size,
                                                       w_init=w0, b_init=b0, seed=seed),
    }

    for r in results.values():
        r["w_real"], r["b_real"] = denormalize_params(r["w"], r["b"], x_mean, x_std)
        r["final_loss"] = r["loss_history"][-1]

    return results, (area, price)


def build_comparison_table(results):
    rows = []
    for name, r in results.items():
        rows.append({
            "Algorithm": name,
            "Final Loss (MSE)": round(r["final_loss"], 2),
            "Updates": r["n_updates"],
            "Training Time (s)": round(r["time_sec"], 5),
            "Fitted w (price/sqft)": round(r["w_real"], 3),
            "Fitted b (intercept)": round(r["b_real"], 2),
        })
    return pd.DataFrame(rows)


def plot_convergence(results, save_path):
    """
    Two panels:
      Left  - full training curve (log scale) so all 3 methods are
              visible on one axis despite very different loss magnitudes
              at the start.
      Right - zoomed view of the LAST few epochs, which is where SGD's
              characteristic noisy/fluctuating convergence (vs Batch GD's
              smooth settle) actually becomes visible.
    """
    colors = {"Batch GD": "tab:blue", "SGD": "tab:orange", "Mini-Batch GD": "tab:green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, r in results.items():
        n_updates = len(r["loss_history"])
        epochs_axis = np.arange(n_updates) / (n_updates / EPOCHS_FOR_PLOT)
        ax1.plot(epochs_axis, r["loss_history"], label=name,
                  linewidth=1.3, color=colors[name])

        # last 20% of training, zoomed in on linear scale
        tail_start = int(n_updates * 0.8)
        ax2.plot(epochs_axis[tail_start:], r["loss_history"][tail_start:],
                  label=name, linewidth=1.1, color=colors[name], alpha=0.85)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (log scale)")
    ax1.set_title("Full convergence")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Zoomed: last 20% of training\n(shows SGD noise vs Batch GD smoothness)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Loss vs Iterations: Batch GD vs SGD vs Mini-Batch GD", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


EPOCHS_FOR_PLOT = 50  # keep in sync with run_all's default epochs


if __name__ == "__main__":
    results, data = run_all(epochs=EPOCHS_FOR_PLOT)

    table = build_comparison_table(results)
    print("\n=== Comparison Table ===")
    print(table.to_string(index=False))

    table.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)
    plot_convergence(results, os.path.join(RESULTS_DIR, "loss_vs_iterations.png"))

    print(f"\nSaved: {RESULTS_DIR}/comparison_table.csv")
    print(f"Saved: {RESULTS_DIR}/loss_vs_iterations.png")
