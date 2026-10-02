"""
experiment.py
-------------
Runs Batch GD, SGD, and Mini-Batch GD on the SAME dataset, model,
loss function, learning rate, and initial parameters.

Compares:
    - final MSE
    - number of parameter updates
    - training time
    - convergence behaviour

Run:
    python experiment.py

Produces:
    results/loss_vs_iterations.png
    results/comparison_table.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gradient_descent import (
    generate_house_data,
    normalize,
    mse_loss,
    batch_gradient_descent,
    stochastic_gradient_descent,
    mini_batch_gradient_descent,
    denormalize_params,
)


# ---------------------------------------------------------------------
# Results folder
# ---------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Run all three algorithms
# ---------------------------------------------------------------------

def run_all(
    n_samples=200,
    epochs=50,
    batch_size=16,
    learning_rate=0.05,
    seed=42
):
    """
    Run Batch GD, SGD, and Mini-Batch GD using the same:

    - dataset
    - model
    - loss function
    - learning rate
    - initial parameters

    Only the number of samples used per update changes.
    """

    # Generate one dataset.
    area, price = generate_house_data(
        n_samples=n_samples,
        seed=seed
    )

    # Normalize both feature and target.
    x_norm, x_mean, x_std = normalize(area)
    y_norm, y_mean, y_std = normalize(price)

    # Same settings for all algorithms.
    common = {
        "lr": learning_rate,
        "epochs": epochs,
        "w_init": 0.0,
        "b_init": 0.0,
    }

    # ---------------------------------------------------------------
    # Train all three algorithms
    # ---------------------------------------------------------------

    results = {

        "Batch GD": batch_gradient_descent(
            x_norm,
            y_norm,
            **common
        ),

        "SGD": stochastic_gradient_descent(
            x_norm,
            y_norm,
            seed=seed,
            **common
        ),

        "Mini-Batch GD": mini_batch_gradient_descent(
            x_norm,
            y_norm,
            batch_size=batch_size,
            seed=seed,
            **common
        ),
    }

    # ---------------------------------------------------------------
    # Convert parameters back to original units
    # ---------------------------------------------------------------

    for result in results.values():

        result["w_real"], result["b_real"] = denormalize_params(
            result["w"],
            result["b"],
            x_mean,
            x_std,
            y_mean,
            y_std
        )

        # Calculate final MSE using the original data.
        result["final_loss"] = mse_loss(
            area,
            price,
            result["w_real"],
            result["b_real"]
        )

    return results, (area, price)


# ---------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------

def build_comparison_table(results):

    rows = []

    for name, result in results.items():

        rows.append({
            "Algorithm": name,
            "Final Loss (MSE)": round(
                result["final_loss"], 2
            ),
            "Updates": result["n_updates"],
            "Training Time (s)": round(
                result["time_sec"], 5
            ),
            "Fitted w (price/sqft)": round(
                result["w_real"], 3
            ),
            "Fitted b (intercept)": round(
                result["b_real"], 2
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------

def plot_convergence(
    results,
    epochs,
    save_path
):
    """
    Plot loss history for all three algorithms.

    Left:
        Full convergence on a logarithmic scale.

    Right:
        Final 20% of training to make SGD's fluctuations easier to see.
    """

    colors = {
        "Batch GD": "tab:blue",
        "SGD": "tab:orange",
        "Mini-Batch GD": "tab:green",
    }

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    for name, result in results.items():

        loss_history = result["loss_history"]

        n_updates = len(loss_history)

        # Convert update number into an approximate epoch position.
        epoch_axis = np.linspace(
            1,
            epochs,
            n_updates
        )

        # Full convergence.
        ax1.plot(
            epoch_axis,
            loss_history,
            label=name,
            linewidth=1.3,
            color=colors[name]
        )

        # Final 20%.
        tail_start = int(n_updates * 0.8)

        ax2.plot(
            epoch_axis[tail_start:],
            loss_history[tail_start:],
            label=name,
            linewidth=1.1,
            color=colors[name],
            alpha=0.85
        )

    # Full convergence graph.
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (normalized scale)")
    ax1.set_title("Full convergence")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    # Zoomed graph.
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss (normalized scale)")
    ax2.set_title("Final 20% of training")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Batch GD vs SGD vs Mini-Batch GD",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.close()


# ---------------------------------------------------------------------
# Run experiment directly
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Experiment settings.
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.05
    SEED = 42

    results, data = run_all(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED
    )

    # Build comparison table.
    table = build_comparison_table(results)

    print("\n=== Comparison Table ===")
    print(
        table.to_string(index=False)
    )

    # Save table.
    table_path = os.path.join(
        RESULTS_DIR,
        "comparison_table.csv"
    )

    table.to_csv(
        table_path,
        index=False
    )

    # Save convergence graph.
    plot_path = os.path.join(
        RESULTS_DIR,
        "loss_vs_iterations.png"
    )

    plot_convergence(
        results,
        EPOCHS,
        plot_path
    )

    print(
        f"\nSaved: {table_path}"
    )
    print(
        f"Saved: {plot_path}"
    )