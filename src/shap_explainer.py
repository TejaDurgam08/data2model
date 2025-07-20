import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_model(model, X_sample, model_name):
    """Returns SHAP bar plot figure for given model and sample input."""
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        # fig = shap.plots.bar(shap_values, show=False)
        # return fig
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False) 
        return fig
    except Exception as e:
        print("heuu")
        print(f"[SHAP Error] {model_name}: {e}")
        return None
