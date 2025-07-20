
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(2, 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    return fig

def plot_predicted_vs_actual(y_test, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    return fig

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    return fig
