import argparse
import os
import logging
import numpy as np
import shutil
from tqdm import tqdm
from src.utils.common import read_yaml, create_directories, get_df
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


STAGE = "Featurization" ## <<< change stage name 

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
  
    ## creating path to train and test directories
    artifacts = config["artifacts"]
    prepare_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    train_data_path = os.path.join(prepare_data_dir_path,artifacts["TRAIN_DATA"])
    test_data_path = os.path.join(prepare_data_dir_path,artifacts["TEST_DATA"])

    # creating directory to store featurized data
    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    create_directories([featurized_data_dir_path])

    # creating path to store to store train.pkl and test.pkl
    featurized_train_dir_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATA_TRAIN"])
    featurized_test_dir_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_DATA_TEST"])


    # parameters path setup
    max_features = params["featurize"]["max_features"]
    n_grams =  params["featurize"]["n_grams"]

    # TRAIN DATA
    # convert data into dataframe
    df_train = get_df(train_data_path)
    # convert sentences to lower case
    train_words = np.array(df_train.text.str.lower().values.astype("U"))

    # create a bag of word vectorizer
    bag_of_words = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1,  n_grams)
    )

    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)

    # creating TFIDF vectorizer on top of BOW 
    tfidf = TfidfTransformer(smooth_idf = False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    # TEST DATA
    # convert data into dataframe
    df_test = get_df(test_data_path)
    # convert sentences to lower case
    test_words = np.array(df_test.text.str.lower().values.astype("U"))

    # create a bag of word vectorizer
    test_words_binary_matrix = bag_of_words.transform(test_words)
    # creating TFIDF vectorizer on top of BOW 
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)

    # call a function to save this matrix

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