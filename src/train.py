
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd

def train_model(df, target_col, model, model_name, task_type, test_size):
    # One-hot encode
    X = pd.get_dummies(df.drop(columns=[target_col]))
    y = df[target_col]
    X, y = X.align(y, axis=0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )
    
    if task_type == "Classification":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred) * 100
        }

    elif task_type == "Regression" and "SVR" in model_name:
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_transformed = sc_y.transform(y_test.values.reshape(-1, 1)).ravel()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        } if task_type == "Regression" else {
            "Accuracy": accuracy_score(y_test, y_pred) * 100
        }

    return model, y_test, y_pred, metrics, X_test
