import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('.artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing input data...")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 40],
                    'min_samples_split': [2, 5, 10, 15]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [20, 50, 100],
                    'subsample':[0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                },
                "CatBoost": {
                    'iterations': [30, 50, 100],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = 0.0
            best_model_name = None
            best_model = None

            for model_name, report in model_report.items():
                if report['r2_score'] > best_model_score:
                    best_model_score = report['r2_score']
                    best_model_name = model_name
                    best_model = report['model']

            if best_model_score < 0.6:
                raise CustomException("No model was found with sufficient accuracy.")
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            
            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)