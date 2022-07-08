import argparse
from cProfile import label
import os
import logging
import shutil
from statistics import mode
from tqdm import tqdm
from src.utils import read_yaml, create_directories
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

STAGE = "STAGE_NAME" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ## Loading data matrix
    ## creating path to train and test directories
    artifacts = config["artifacts"]
    # establishing path to our featurized data
    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    featurized_train_dir_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATA_TRAIN"])
    featurized_test_dir_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATA_TEST"])

    # creating a model directory
    model_dir = artifacts["MODEL_DIR"]
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], model_dir)
    create_directories([model_dir_path])

    # create the path to store model
    model_name = artifacts["MODEL_NAME"]
    model_path = os.path.join(model_dir_path, model_name)

    matrix = joblib.load(featurized_train_dir_path)

    labels = np.squeeze(matrix[:, 1].toarray()) # from matrix take all the rows and 1st column(Text)and convert it to array
    X = matrix[:, 2:] # get training data from 2nd row onwards

    logging.info(f"input matrix size: {matrix.shape}")
    logging.info(f"input matrix size: {X.shape}")
    logging.info(f"input matrix size: {labels.shape}")

    seed = params["train"]["seed"]
    n_est = params["train"]["n_est"]
    n_jobs = params["train"]["n_jobs"]
    min_split = params["train"]["min_split"]

    model = RandomForestClassifier(
        n_estimators=n_est,
        min_samples_split=min_split,
        n_jobs=n_jobs,
        random_state=seed
    )

    model.fit(X, labels)

    joblib.dump(model, model_path)
    logging.info(f"saved the model: {model_path}")



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e