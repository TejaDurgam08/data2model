import streamlit as st
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, accuracy_score



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)


from visualizer import (
    plot_confusion_matrix,
    plot_predicted_vs_actual,
    plot_residuals
)
from shap_explainer import explain_model
from report_generator import generate_report


st.set_page_config(page_title="Results & Visualization", layout="wide")
st.title("📈Results & Visualization")

# Check if results exist
if "results" not in st.session_state or not st.session_state["results"]:
    st.warning("⚠️ No trained models found. Please complete Step 2 first.")
    st.stop()

# Select which model result to view
model_names = [r["name"] for r in st.session_state["results"]]
selected = st.selectbox("Select model to visualize", model_names)

# Find selected result
selected_result = next(r for r in st.session_state["results"] if r["name"] == selected)

y_test = selected_result["y_test"]
y_pred = selected_result["y_pred"]
task_type = selected_result["task"]

if task_type == "Classification":
    st.subheader("🧮 Confusion Matrix")
    fig = plot_confusion_matrix(y_test, y_pred)
    st.pyplot(fig)

elif task_type == "Regression":
    st.subheader("📊 Predicted vs Actual")
    fig = plot_predicted_vs_actual(y_test, y_pred)
    st.pyplot(fig)

    st.subheader("📉 Residual Plot")
    fig = plot_residuals(y_test, y_pred)
    st.pyplot(fig)


st.subheader("📉 SHAP Explanation")

model_obj = selected_result["model"]
X_sample = st.session_state.get("last_X_test", None)

if X_sample is not None:
    fig = explain_model(model_obj, X_sample, selected)
    if fig:
        st.pyplot(fig)
    else:
        st.info("SHAP explanation not supported for this model.")
else:
    st.info("No input data saved for SHAP. Please retrain model.")


st.markdown("---")
st.subheader("🏆 Model Comparison Summary")

metrics_summary = []

for res in st.session_state["results"]:
    model_name = res["name"]
    y_test = res["y_test"]
    y_pred = res["y_pred"]
    task = res["task"]

    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        metrics_summary.append({
            "Model": model_name,
            "Task": task,
            "Accuracy (%)": round(acc * 100, 2),
            "R² Score": None,
            "MSE": None
        })
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        metrics_summary.append({
            "Model": model_name,
            "Task": task,
            "Accuracy (%)": None,
            "R² Score": round(r2, 4),
            "MSE": round(mse, 4)
        })

# Show summary table
summary_df = pd.DataFrame(metrics_summary)
st.dataframe(summary_df)

# Plot chart
st.subheader("📊 Performance Comparison")
chart_metric = st.selectbox("Choose metric to compare", ["Accuracy (%)", "R² Score", "MSE"])
filtered_df = summary_df.dropna(subset=[chart_metric])

if not filtered_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=filtered_df, x="Model", y=chart_metric, hue="Task", ax=ax)
    ax.set_title(f"{chart_metric} by Model")
    st.pyplot(fig)
else:
    st.info("No models available for this metric.")


if st.button("📤 Generate PDF Report"):
    st.info("📄 Generating report... Please wait.")
    pdf_path = generate_report(st.session_state["results"])

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Report",
            data=f,
            file_name="ml_model_report.pdf",
            mime="application/pdf"
        )