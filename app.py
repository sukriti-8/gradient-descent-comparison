"""
app.py
------
Streamlit UI for comparing:
Batch Gradient Descent,
Stochastic Gradient Descent (SGD),
and Mini-Batch Gradient Descent.

The three algorithms are trained on the same:
- dataset
- Linear Regression model
- MSE loss
- learning rate
- initial parameters
- number of epochs

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from gradient_descent import (
    generate_house_data,
    normalize,
    denormalize_params,
    predict,
    batch_gradient_descent,
    stochastic_gradient_descent,
    mini_batch_gradient_descent,
)


# ==============================================================
# PAGE SETUP
# ==============================================================

st.set_page_config(
    page_title="Gradient Descent Comparison",
    page_icon="📉",
    layout="wide"
)


# ==============================================================
# TITLE
# ==============================================================

st.title("📉 Batch GD vs SGD vs Mini-Batch GD")

st.write(
    "Compare three Gradient Descent variants on the same "
    "Linear Regression problem: house area → house price."
)

st.caption(
    "Controlled experiment: same dataset, model, MSE loss, "
    "learning rate and initial parameters."
)


# ==============================================================
# SIDEBAR
# ==============================================================

st.sidebar.header("⚙️ Experiment Settings")


# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------

st.sidebar.subheader("Dataset")

dataset_type = st.sidebar.radio(
    "Choose dataset",
    [
        "Synthetic House Data",
        "Upload CSV"
    ]
)


# --------------------------------------------------------------
# Synthetic dataset settings
# --------------------------------------------------------------

if dataset_type == "Synthetic House Data":

    n_samples = st.sidebar.slider(
        "Number of samples",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )

    noise = st.sidebar.slider(
        "Noise level",
        min_value=0.0,
        max_value=50000.0,
        value=15000.0,
        step=1000.0
    )

    uploaded_file = None

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="First column = input X, second column = target Y"
    )

    n_samples = 200
    noise = 15000.0


# --------------------------------------------------------------
# Training settings
# --------------------------------------------------------------

st.sidebar.subheader("Training")

epochs = st.sidebar.slider(
    "Epochs",
    min_value=5,
    max_value=200,
    value=50,
    step=5
)

batch_size = st.sidebar.slider(
    "Mini-Batch Size",
    min_value=2,
    max_value=64,
    value=16,
    step=2
)

learning_rate = st.sidebar.number_input(
    "Common Learning Rate",
    min_value=0.0001,
    max_value=5.0,
    value=0.05,
    step=0.01,
    format="%.4f"
)


run_button = st.sidebar.button(
    "🚀 Run Comparison",
    type="primary",
    use_container_width=True
)


# ==============================================================
# LOAD DATA
# ==============================================================

def load_data():

    if dataset_type == "Upload CSV":

        if uploaded_file is None:
            st.warning("Please upload a CSV file.")
            st.stop()

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the CSV file: {e}")
            st.stop()

        if df.shape[1] < 2:
            st.error(
                "CSV must contain at least two columns."
            )
            st.stop()

        # Use first two columns
        x = pd.to_numeric(
            df.iloc[:, 0],
            errors="coerce"
        ).to_numpy()

        y = pd.to_numeric(
            df.iloc[:, 1],
            errors="coerce"
        ).to_numpy()

        # Remove invalid rows
        valid = np.isfinite(x) & np.isfinite(y)

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            st.error(
                "The dataset must contain at least two valid numeric samples."
            )
            st.stop()

        return x, y

    else:

        x, y = generate_house_data(
            n_samples=n_samples,
            noise=noise,
            seed=42
        )

        return x, y


# ==============================================================
# RUN EXPERIMENT
# ==============================================================

if run_button:

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------

    x_raw, y_raw = load_data()

    if len(x_raw) < 2:
        st.error(
            "The dataset must contain at least two samples."
        )
        st.stop()

    if np.std(x_raw) < 1e-12:
        st.error(
            "The input values must not all be identical."
        )
        st.stop()

    if np.std(y_raw) < 1e-12:
        st.error(
            "The target values must not all be identical."
        )
        st.stop()


    # ----------------------------------------------------------
    # Normalize data
    # ----------------------------------------------------------

    # Normalization makes gradient calculations stable.
    x_norm, x_mean, x_std = normalize(x_raw)
    y_norm, y_mean, y_std = normalize(y_raw)


    # ----------------------------------------------------------
    # Train algorithms
    # ----------------------------------------------------------

    with st.spinner("Training the three algorithms..."):

        # ------------------------------------------------------
        # Batch Gradient Descent
        # ------------------------------------------------------

        batch_result = batch_gradient_descent(
            x_norm,
            y_norm,
            lr=learning_rate,
            epochs=epochs,
            w_init=0.0,
            b_init=0.0
        )


        # ------------------------------------------------------
        # Stochastic Gradient Descent
        # ------------------------------------------------------

        sgd_result = stochastic_gradient_descent(
            x_norm,
            y_norm,
            lr=learning_rate,
            epochs=epochs,
            w_init=0.0,
            b_init=0.0,
            seed=42
        )


        # ------------------------------------------------------
        # Mini-Batch Gradient Descent
        # ------------------------------------------------------

        minibatch_result = mini_batch_gradient_descent(
            x_norm,
            y_norm,
            lr=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            w_init=0.0,
            b_init=0.0,
            seed=42
        )


    # ----------------------------------------------------------
    # Store results
    # ----------------------------------------------------------

    results = {
        "Batch GD": batch_result,
        "SGD": sgd_result,
        "Mini-Batch GD": minibatch_result
    }


    # ==========================================================
    # CONVERT PARAMETERS BACK TO ORIGINAL UNITS
    # ==========================================================

    for result in results.values():

        result["w_real"], result["b_real"] = denormalize_params(
            result["w"],
            result["b"],
            x_mean,
            x_std,
            y_mean,
            y_std
        )

        # Predictions using original X values
        predictions = predict(
            x_raw,
            result["w_real"],
            result["b_real"]
        )

        # Final MSE using original price units
        result["final_loss"] = np.mean(
            (predictions - y_raw) ** 2
        )


    # ==========================================================
    # SUCCESS MESSAGE
    # ==========================================================

    st.success("Training completed successfully!")


    # ==========================================================
    # COMPARISON RESULTS
    # ==========================================================

    st.header("📊 Comparison Results")

    st.caption(
        "Final MSE represents the overall squared prediction error "
        "across the complete dataset."
    )


    comparison_data = []

    for name, result in results.items():

        comparison_data.append(
            {
                "Algorithm": name,

                "Final MSE": round(
                    result["final_loss"],
                    2
                ),

                "Updates": result["n_updates"],

                "Training Time (s)": round(
                    result["time_sec"],
                    5
                ),

                "Slope (w)": round(
                    result["w_real"],
                    3
                ),

                "Intercept (b)": round(
                    result["b_real"],
                    2
                )
            }
        )


    comparison_table = pd.DataFrame(
        comparison_data
    )


    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True
    )


    # ----------------------------------------------------------
    # Download results
    # ----------------------------------------------------------

    csv_data = comparison_table.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_data,
        file_name="comparison_results.csv",
        mime="text/csv"
    )


    # ==========================================================
    # CONVERGENCE GRAPH
    # ==========================================================

    st.header("📈 Convergence Comparison")

    st.write(
        "This graph shows how the MSE loss changes during training. "
        "The loss is calculated on normalized data so that the three "
        "optimization methods can be compared on a stable scale."
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    for name, result in results.items():

        loss_history = result["loss_history"]

        number_of_updates = len(
            loss_history
        )

        # Map individual updates to the epoch range
        epoch_axis = np.linspace(
            1,
            epochs,
            number_of_updates
        )

        ax.plot(
            epoch_axis,
            loss_history,
            label=name,
            linewidth=1.5
        )


    ax.set_yscale("log")

    ax.set_xlabel("Epoch")

    ax.set_ylabel(
        "MSE Loss (log scale)"
    )

    ax.set_title(
        "Loss Convergence of the Three Algorithms"
    )

    ax.legend()

    ax.grid(
        alpha=0.3,
        which="both"
    )

    st.pyplot(fig)

    plt.close(fig)


    # ==========================================================
    # FITTED REGRESSION LINES
    # ==========================================================

    st.header("📐 Fitted Regression Lines")

    st.write(
        "The points represent the actual dataset. "
        "Each line represents the Linear Regression model "
        "learned by one optimization method."
    )


    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )


    # ----------------------------------------------------------
    # Actual data
    # ----------------------------------------------------------

    ax2.scatter(
        x_raw,
        y_raw,
        s=15,
        alpha=0.4,
        label="Actual Data"
    )


    # ----------------------------------------------------------
    # Smooth X values
    # ----------------------------------------------------------

    x_line = np.linspace(
        x_raw.min(),
        x_raw.max(),
        100
    )


    # ----------------------------------------------------------
    # Regression lines
    # ----------------------------------------------------------

    for name, result in results.items():

        y_line = predict(
            x_line,
            result["w_real"],
            result["b_real"]
        )

        ax2.plot(
            x_line,
            y_line,
            linewidth=2,
            label=name
        )


    ax2.set_xlabel(
        "House Area (sq ft)"
    )

    ax2.set_ylabel(
        "House Price"
    )

    ax2.set_title(
        "Regression Lines Learned by Each Algorithm"
    )

    ax2.legend()

    ax2.grid(
        alpha=0.3
    )


    st.pyplot(fig2)

    plt.close(fig2)


    # ==========================================================
    # KEY OBSERVATIONS
    # ==========================================================

    st.header("🔍 Key Observations")


    st.markdown(
        f"""
### Batch Gradient Descent

- Uses the **complete dataset** for every parameter update.
- Performs **{batch_result["n_updates"]} updates** for {epochs} epochs.
- Its updates are generally smooth because each update uses all samples.

### Stochastic Gradient Descent

- Uses **one training sample** for each parameter update.
- Performs **{sgd_result["n_updates"]} updates** for {epochs} epochs.
- Its loss can fluctuate because each update is based on one sample.

### Mini-Batch Gradient Descent

- Uses a **small group of samples** for each update.
- With a batch size of **{batch_size}**, it performs
  **{minibatch_result["n_updates"]} updates**.
- It provides a middle approach between Batch GD and SGD.

### Experimental Setup

All three algorithms use:

- Same dataset
- Same Linear Regression model
- Same MSE loss
- Same learning rate: **{learning_rate}**
- Same number of epochs: **{epochs}**
- Same initial parameters: **w = 0, b = 0**

### Important Interpretation

The **Final MSE** measures the overall prediction error of the
learned model across the dataset. A lower MSE means lower overall
squared error for that particular experiment.

It does **not** mean that the algorithm must produce the closest
prediction for every individual data point.
"""
    )


# ==============================================================
# INITIAL SCREEN
# ==============================================================

else:

    st.info(
        "Choose the experiment settings from the sidebar "
        "and click **🚀 Run Comparison**."
    )


    st.markdown(
        """
### What this project compares

| Algorithm | Samples used per update |
|---|---:|
| Batch GD | All training samples |
| SGD | 1 sample |
| Mini-Batch GD | Small batch |

The goal is to observe how the amount of data used in each
gradient update affects:

- **Convergence behaviour**
- **Number of updates**
- **Training time**
- **Final model fit**
"""
    )