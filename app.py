"""
app.py
------
Streamlit UI for the Comparative Study of Batch GD, SGD, and Mini-Batch GD.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from gradient_descent import (
    generate_house_data, normalize, denormalize_params, predict,
    batch_gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent,
)

st.set_page_config(page_title="GD Variants Comparison", layout="wide")

st.title("📉 Batch GD vs SGD vs Mini-Batch GD")
st.caption(
    "A live comparison of the three Gradient Descent variants on a simple "
    "Linear Regression problem (house area → house price)."
)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

data_source = st.sidebar.radio("Dataset", ["Synthetic (house area vs price)", "Upload CSV"])

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "CSV with two numeric columns: first = x (feature), second = y (target)",
        type=["csv"],
    )
else:
    uploaded = None
    n_samples = st.sidebar.slider("Number of samples", 50, 1000, 200, step=50)
    noise = st.sidebar.slider("Noise level", 0.0, 50000.0, 15000.0, step=1000.0)

st.sidebar.subheader("Training")
epochs = st.sidebar.slider("Epochs", 5, 200, 50, step=5)
batch_size = st.sidebar.slider("Mini-batch size", 2, 128, 16, step=2)

st.sidebar.subheader("Learning rates")
lr_batch = st.sidebar.number_input("Batch GD learning rate", 0.0001, 5.0, 0.05, step=0.01, format="%.4f")
lr_sgd = st.sidebar.number_input("SGD learning rate", 0.0001, 5.0, 0.01, step=0.005, format="%.4f")
lr_minibatch = st.sidebar.number_input("Mini-Batch GD learning rate", 0.0001, 5.0, 0.03, step=0.01, format="%.4f")

run_button = st.sidebar.button("🚀 Run Comparison", type="primary")

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data():
    if data_source == "Upload CSV" and uploaded is not None:
        df = pd.read_csv(uploaded)
        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
        return x, y
    x, y = generate_house_data(n_samples=n_samples, noise=noise)
    return x, y


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if run_button:
    if data_source == "Upload CSV" and uploaded is None:
        st.warning("Please upload a CSV file first, or switch to the synthetic dataset.")
        st.stop()

    x_raw, y = load_data()
    x_norm, x_mean, x_std = normalize(x_raw)

    with st.spinner("Training all three algorithms..."):
        results = {
            "Batch GD": batch_gradient_descent(x_norm, y, lr=lr_batch, epochs=epochs),
            "SGD": stochastic_gradient_descent(x_norm, y, lr=lr_sgd, epochs=epochs),
            "Mini-Batch GD": mini_batch_gradient_descent(
                x_norm, y, lr=lr_minibatch, epochs=epochs, batch_size=batch_size
            ),
        }
        for r in results.values():
            r["w_real"], r["b_real"] = denormalize_params(r["w"], r["b"], x_mean, x_std)
            r["final_loss"] = r["loss_history"][-1]

    st.success("Done! Results below.")

    # --- Comparison table -------------------------------------------------
    st.subheader("📊 Comparison Table")
    table = pd.DataFrame([
        {
            "Algorithm": name,
            "Final Loss (MSE)": f'{r["final_loss"]:.2f}',
            "Updates": r["n_updates"],
            "Training Time (s)": f'{r["time_sec"]:.5f}',
            "Fitted slope (w)": f'{r["w_real"]:.3f}',
            "Fitted intercept (b)": f'{r["b_real"]:.2f}',
        }
        for name, r in results.items()
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode()
    st.download_button("⬇️ Download comparison table (CSV)", csv_bytes, "comparison_table.csv", "text/csv")

    # --- Convergence plot ---------------------------------------------------
    st.subheader("📈 Loss vs Iterations")
    colors = {"Batch GD": "tab:blue", "SGD": "tab:orange", "Mini-Batch GD": "tab:green"}
    col1, col2 = st.columns(2)

    fig1, ax1 = plt.subplots(figsize=(6, 4.2))
    for name, r in results.items():
        n_updates = len(r["loss_history"])
        epochs_axis = np.arange(n_updates) / (n_updates / epochs)
        ax1.plot(epochs_axis, r["loss_history"], label=name, color=colors[name], linewidth=1.3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (log scale)")
    ax1.set_title("Full convergence")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    for name, r in results.items():
        n_updates = len(r["loss_history"])
        epochs_axis = np.arange(n_updates) / (n_updates / epochs)
        tail_start = int(n_updates * 0.8)
        ax2.plot(epochs_axis[tail_start:], r["loss_history"][tail_start:],
                 label=name, color=colors[name], linewidth=1.1, alpha=0.85)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Zoomed: last 20% of training")
    ax2.legend()
    ax2.grid(alpha=0.3)
    col2.pyplot(fig2)

    # --- Fitted line vs data --------------------------------------------
    st.subheader("📐 Fitted Line vs Data")
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.scatter(x_raw, y, s=12, alpha=0.4, color="gray", label="Data")
    x_line = np.linspace(x_raw.min(), x_raw.max(), 100)
    for name, r in results.items():
        ax3.plot(x_line, predict(x_line, r["w_real"], r["b_real"]),
                  label=f"{name} fit", color=colors[name], linewidth=2)
    ax3.set_xlabel("x (feature)")
    ax3.set_ylabel("y (target)")
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)

    # --- Observations -----------------------------------------------------
    st.subheader("🔍 Observations")
    st.markdown(f"""
    - **Batch GD** made **{results['Batch GD']['n_updates']}** update(s) (one per epoch) and converged the most **smoothly**.
    - **SGD** made **{results['SGD']['n_updates']}** updates and converged the **fastest in wall-clock terms per epoch**, but shows visible **fluctuation/noise** near the minimum.
    - **Mini-Batch GD** made **{results['Mini-Batch GD']['n_updates']}** updates — a **balance** between the stability of Batch GD and the speed of SGD.
    """)
else:
    st.info("Set your options in the sidebar and click **🚀 Run Comparison** to begin.")
