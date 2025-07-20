from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def get_models(task_type):
    if task_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "SVR": SVR(),
            "Random Forest Regressor": RandomForestRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
    elif task_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(),
            "SVC": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "KNN Classifier": KNeighborsClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier()
        }
    else:
        return {}
