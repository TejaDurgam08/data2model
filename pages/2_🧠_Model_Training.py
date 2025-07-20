
import streamlit as st
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from train import train_model
from model_selector import get_models
from model_saver import save_model



st.set_page_config(page_title="Model Selection & Training", layout="wide")
st.title("🧠 Step 2: Select & Train ML Models")

# Check for cleaned data
if "processed_df" not in st.session_state:
    st.warning("⚠️ Please complete Step 1: Upload and clean your dataset first.")
    st.stop()

df = st.session_state["processed_df"]


st.markdown("### 🎯 Select Target, Task, and Models")

col1, col2, col3 = st.columns(3)

with col1:
    target_col = st.selectbox("Target column", options=df.columns)

with col2:
    task_type = st.radio("Task type", ["Regression", "Classification"])

with col3:
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5)

available_models = get_models(task_type)
selected_model_names = st.multiselect("🧠 Choose models to train", options=list(available_models.keys()))

# Prepare data
X = df.drop(columns=[target_col])
y = df[target_col]
X = pd.get_dummies(X)
X, y = X.align(y, axis=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, random_state=42
)

st.subheader("✅ Selected Models")
if selected_model_names:
    st.markdown(", ".join([f"🧠 **{name}**" for name in selected_model_names]))
else:
    st.info("No models selected yet.")


# Train and evaluate
if st.button("🚀 Train Models"):
    st.subheader("📊 Results")

    if not selected_model_names:
        st.warning("Please select at least one model to train.")
        st.stop()

    if "results" not in st.session_state:
        st.session_state["results"] = []

    for model_name in selected_model_names:
        st.markdown(f"🔄 **Training model: {model_name}...**")
        st.markdown("⏳ **This may take some time...** Please wait while the models are being trained.")


        model = available_models[model_name]

        model, y_test, y_pred, metrics, x_test = train_model(df, target_col, model, model_name, task_type, test_size )
        # Save X_test for SHAP 
        st.session_state["last_X_test"] = pd.DataFrame(X_test).iloc[:50]

        # Display status
        st.markdown(f"✅ **Completed: {model_name}**")

        # Show metrics
        st.markdown(f"## 🔍 {model_name}")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v:.4f}")

        # Save results
        st.session_state["results"].append({
            "name": model_name,
            "task": task_type,
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred
        })

        model_path = save_model(model, model_name)
        st.success(f"Model **{model_name}** saved")

        # download button
        with open(model_path, "rb") as f:
            st.download_button(
                label=f"📥 Download {model_name}",
                data=f,
                file_name=f"{model_name}.joblib"
            )
