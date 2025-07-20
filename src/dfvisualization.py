import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def x_vs_y_plot(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"📈 {x_col} vs {y_col}")
    return fig


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("🔗 Feature Correlation Heatmap")
    return fig

def plot_feature_distributions(df):
    num_cols = df.select_dtypes(include='number').columns
    fig, axes = plt.subplots(nrows=len(num_cols), ncols=1, figsize=(8, len(num_cols)*2.5))
    if len(num_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"📊 Distribution of '{col}'")
    fig.tight_layout()
    return fig

def plot_pairplot(df):
    plot_df = df.copy()
    fig = sns.pairplot(plot_df)
    return fig
