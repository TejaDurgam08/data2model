
import streamlit as st
import pandas as pd
import os
import sys

# Import from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from data_loader import load_data
from preprocess import handle_missing_values
from dfvisualization import plot_correlation_heatmap, plot_feature_distributions, plot_pairplot,x_vs_y_plot

st.set_page_config(page_title="Data Upload & Cleaning", layout="wide")
st.title("📂 Upload and Clean Your Dataset")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])


if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.success("✅ File loaded successfully!")


        st.sidebar.header("1. Handle Missing Values")

        # Check if there are any NaN values
        total_missing = df.isnull().sum().sum()

        if total_missing == 0:
            st.sidebar.success("✅ No missing values found in the dataset.")
            nan_handling_method = "None"
            fill_value = None
            processed_df = df  # Nothing to change
        else:
            nan_handling_method = st.sidebar.selectbox("Choose a method", [
                "None", "Drop rows", "Drop columns",
                "Fill with mean", "Fill with median", "Fill with mode",
                "Fill with constant", "Interpolate"
            ])

            fill_value = None
            if nan_handling_method == "Fill with constant":
                fill_value = st.sidebar.text_input("Enter constant value to fill NaNs", value="0")

            # Apply preprocessing
            processed_df = handle_missing_values(df, nan_handling_method, fill_value)

        st.sidebar.header("2. Columns to Drop")

        columns_to_drop = st.sidebar.multiselect("Select columns to drop", options=processed_df.columns.tolist())
        if columns_to_drop:
            processed_df = processed_df.drop(columns=columns_to_drop)
            st.sidebar.success("✅ Columns dropped successfully!")

        
       #Dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        # Horizontal layout with 4 columns
        col1, col2, col3 = st.columns(3)

        # Column 1: Shape info
        with col1:
            st.markdown("#### 📐 Shape")
            st.markdown(f"- **Rows:** {df.shape[0]}")
            st.markdown(f"- **Columns:** {df.shape[1]}")

        # Column 2: Column Types
        with col2:
            st.markdown("#### 🧱 Column Types")
            st.write(df.dtypes)

        # Column 3: Missing Values
        with col3:
            st.markdown("#### ❌ Missing Values")
            st.write(df.isnull().sum())

        # Column 4: Basic Statistics
        
        st.markdown("#### 📊 Basic Statistics")
        st.write(df.describe(include='all'))

        

        

        # Show processed dataset
        st.subheader("🧹 Processed Dataset")
        st.dataframe(processed_df.head(), use_container_width=True)
        st.markdown(f"- **Rows:** {processed_df.shape[0]}")
        st.markdown(f"- **Columns:** {processed_df.shape[1]}")

        st.session_state["processed_df"] = processed_df

        #saving processed data
        save_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())



        st.markdown("---")
        st.subheader("🔍 Data Exploration")

        st.markdown("Use the options below to visualize your data.")

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column for scatter plot", options=processed_df.columns)
        with col2:
            y_col = st.selectbox("Select Y-axis column for scatter plot", options=processed_df.columns)

        if x_col and y_col:
            fig = x_vs_y_plot(processed_df, x_col, y_col)
            st.pyplot(fig)

        if st.checkbox("📊  Show Correlation Heatmap"):
            fig = plot_correlation_heatmap(processed_df)
            st.pyplot(fig)

        if st.checkbox("📊 Show Feature Distributions"):
            fig = plot_feature_distributions(processed_df)
            st.pyplot(fig)

        if st.checkbox("📊  Show Pairplot (slow for big datasets)"):
            #target_col = st.selectbox("Target column for pairplot coloring ", options=processed_df.columns)
            fig = plot_pairplot(processed_df)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.warning("👈 Upload a dataset from the sidebar to get started.")


