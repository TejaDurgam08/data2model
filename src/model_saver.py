import os
import joblib

def save_model(model, name, folder="models"):
    #Save the model to a .joblib file
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, filepath)
    return filepath

def load_model(filepath):
    return joblib.load(filepath)
