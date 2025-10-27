import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
             # Extract metrics from classification_metric
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score
            
            # Log metrics to MLflow
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall", recall_score)
            mlflow.sklearn.log_model(best_model, "best_model")
        
    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            # Define models dictionary
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "Adaboost": AdaBoostClassifier()
            }
            
            # Define params dictionary for hyperparameter tuning
            params = {  
                "Decision Tree": {  
                    'criterion': ['gini', 'entropy', 'log_loss'],  
                },  
                "Random Forest": {  
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },  
                "Gradient Boosting": {  
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],  
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},  
                "Adaboost": {  
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],  
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }  
            }
            
            logging.info("Starting model evaluation with all classifiers")
            
            # Call evaluate_models with all required arguments including params
            model_report: dict = evaluate_models(
                X_train=x_train, 
                y_train=y_train, 
                X_test=x_test, 
                y_test=y_test, 
                models=models,
                params=params  # Add this parameter
            )
            
            # To get best model_score from dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            # Training predictions and metrics
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            # Track the mlflow
            self.track_mlflow(best_model,classification_train_metric)
            # Test predictions and metrics
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            self.track_mlflow(best_model,classification_test_metric)

            # Load the preprocessor
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Create directory for saving the model
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # Create NetworkModel instance
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            ## Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            logging.info(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")
            logging.info(f"Training target unique values: {set(y_train)}")
            
            # Pass all required arguments to train_model
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)