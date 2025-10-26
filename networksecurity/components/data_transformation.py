import sys 
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline 

from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifacts,
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifacts,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifacts = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
   
    def get_data_transformer_object(self) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file 
        and returns a Pipeline object with the KNNImputer object as the first step.

        Returns:
            A Pipeline object
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialised KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor: Pipeline = Pipeline([("imputer", imputer)])
            logging.info("Created preprocessing pipeline with KNNImputer")
            return processor
        except Exception as e:
            logging.error(f"Error in creating data transformer object: {e}")
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        
        try:
            logging.info("Starting data transformation")
            
            # Read train and test data
            train_df = DataTransformation.read_data(
                file_path=self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                file_path=self.data_validation_artifact.valid_test_file_path
            )
            
            logging.info(f"Read train data with shape: {train_df.shape}")
            logging.info(f"Read test data with shape: {test_df.shape}")

            # Training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # Testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            logging.info("Separated features and target variables")
            logging.info(f"Training features shape: {input_feature_train_df.shape}")
            logging.info(f"Testing features shape: {input_feature_test_df.shape}")

            # Get preprocessor and transform data
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            
            logging.info("Applied preprocessing transformation")

            # Combine features and targets
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)] 
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            
            logging.info(f"Combined train array shape: {train_arr.shape}")
            logging.info(f"Combined test array shape: {test_arr.shape}")
            
            # Save numpy array data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, 
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, 
                array=test_arr
            )
            save_object(
                self.data_transformation_config.transformed_object_file_path, 
                preprocessor_object
            )
            
            logging.info("Saved all transformed data and preprocessor object")

            # Create and return the transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data transformation completed successfully")
            logging.info(f"Created artifact: {data_transformation_artifact}")
            
            return data_transformation_artifact
        
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise NetworkSecurityException(e, sys)