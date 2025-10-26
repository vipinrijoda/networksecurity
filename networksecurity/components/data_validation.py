from networksecurity.entity.artifact_entity import DataIngestionArtifacts,DataValidationArtifacts
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifacts, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
            # This line assumes a utility function 'read_yaml_file' exists
            # and 'SCHEMA_FILE_PATH' is defined elsewhere.
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH) 
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Checks if the dataframe has the correct number of columns based on the schema."""
        try:
            # Assumes the number of required columns is the length of the 'columns' section in the schema
            required_columns = len(self._schema_config['columns']) 
            
            # Log required and actual column counts
            logging.info(f"Required number of columns: {required_columns}") 
            logging.info(f"Data frame has columns: {len(dataframe.columns)}") 
            
            # The actual validation logic is missing in the snippet, but would be:
            validation_status = (len(dataframe.columns) == required_columns)
            return validation_status

        except Exception as e:
            raise NetworkSecurityException(e, sys)   

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        """Detects data drift between a base dataframe and the current dataframe."""
        try:
            status = True #
            report = {} #

            for column in base_df.columns: #
                d1 = base_df[column] #
                d2 = current_df[column] #
                
                is_sample_dist = ks_2samp(d1, d2) # Kolmogorov-Smirnov 2-sample test
                
                if threshold <= is_sample_dist.pvalue: #
                    is_found = False #
                else:
                    is_found = True #
                    status = False #

                # Report update logic
                report.update({column:{
                    "p_value":float(is_sample_dist.pvalue),
                    "drift_status":is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)        
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys) #      



    def initiate_data_validation(self) -> DataValidationArtifacts:
        try:
            error_message = ""
        
        # Retrieves the file paths from the previous ingestion step
            train_file_path = self.data_ingestion_artifact.trained_file_path 
            test_file_path = self.data_ingestion_artifact.test_file_path 
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

        ## Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train dataframe does not contain all columns.\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test dataframe does not contain all columns.\n"    
        
        # If there are validation errors, raise exception
            if error_message:
                raise Exception(f"Data Validation Failed: {error_message}")
        
        # Check data drift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
        
        # Create directory for validated data
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

        # Save validated datasets to validation directory
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
           )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
        
        # Return validation file paths
            data_validation_artifact = DataValidationArtifacts(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact 
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
        
