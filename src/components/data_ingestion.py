import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('.artifacts', 'train.csv')
    test_data_path: str = os.path.join('.artifacts', 'test.csv')
    raw_data_path: str = os.path.join('.artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initialize_data_ingestion(self):
        logging.info("Data Ingestion method starts...")
        try:
            logging.info("Reading the dataset...")
            # Read the dataset
            df = pd.read_csv('notebooks/data/StudentsPerformance.csv')
            logging.info("Dataset read as pandas DataFrame.")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder.")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            
            logging.info("Train and test data saved to artifacts folder.")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initialize_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)

    print(f"R2 Score: {r2_score}")



