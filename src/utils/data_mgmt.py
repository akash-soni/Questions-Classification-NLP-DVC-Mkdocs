from cProfile import label
import logging
from random import random
from turtle import heading
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
import re
import joblib
import numpy as np
import scipy.sparse as sparse


def process_post(fd_in, fd_out_train, fd_out_test, target_tag, split):
    line_num = 1
    column_names = "pid\tlabel\ttext\n"
    fd_out_train.write(column_names)
    fd_out_test.write(column_names)

    for line in tqdm(fd_in):
        try:
            # whatever random line is being read it will decide wheather it is train or tesgt
            fd_out = fd_out_train if random.random() > split else fd_out_test
           
            attr = ET.fromstring(line).attrib ## getting the tags

            pid = attr.get('Id', "") # post Id
            label = 1 if target_tag in attr.get('Tags',"") else 0 # if Tags exists not available 0 othrwise 1
            title = re.sub(r"\s+"," ", attr.get('Title', "")).strip() # remove extra spaces from title
            body = re.sub(r"\s+"," ", attr.get('Body', "")).strip() # remove extra spaces from test
            text = f"{title}{body}"

            fd_out.write(f"{pid}\t{label}\t{text}\n")
            line_num += 1
        except Exception as e:
            msg = f"skipping the broken lines {line_num}: {e}\n"
            logging.exception(msg)



def save_matrix(df, text_matrix, out_path):
    pid_matrix = sparse.csr_matrix(df.pid.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([pid_matrix, label_matrix, text_matrix])

    msg = f"the output matrix saved at {out_path} of shape: {result.shape}"
    logging.info(msg)
    joblib.dump(result, out_path )
    

    
