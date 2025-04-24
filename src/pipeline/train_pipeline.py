import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

def main():
    try:
        # Step 1: Data Ingestion
        logging.info("Initializing Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initialize_data_ingestion()
        logging.info(f"Data Ingestion completed.")

        # Step 2: Data Transformation
        logging.info("Initializing Data Transformation...")
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )
        logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        logging.info("Initializing Model Training...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(
            train_array=train_array,
            test_array=test_array,
            preprocessor_path=preprocessor_path
        )
        logging.info(f"Model Training completed. R2 Score: {r2_score}")

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    main()


