import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import joblib

import numpy as np
import pandas as pd

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}

    for model_name, model in models.items():
        try:
            params_grid = params[model_name]
            # print(f"Evaluating model: {model_name}")
            # print(f"Parameters: {params_grid}")
            if params_grid:
                grid_search = GridSearchCV(model, params_grid, cv=5, scoring='r2')
                grid_search.fit(X_train, y_train)
                # print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_test_pred)
            model_report[model_name] = {
                'r2_score': r2,
                'model': model
            }
        except Exception as e:
            raise CustomException(e, sys)

    return model_report

def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)