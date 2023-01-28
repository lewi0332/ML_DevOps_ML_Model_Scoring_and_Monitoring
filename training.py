"""
This script trains a logistic regression model on the data in finaldata.csv
and saves the trained model in a file called trainedmodel.pkl

Author: Derrick Lewis
Date: 2023-01-28
"""
import os
import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Function for training the model
def train_model(data_pth, model_pth):
    """
    Trains a logistic regression model on the data in finaldata.csv

    Parameters
    ---
    data_path: str
        Path to the directory containing finaldata.csv
    model_path: str
        Path to the directory where the trained model will be saved

    Returns
    ---
    None
    """
    dff = pd.read_csv(data_pth + '/finaldata.csv')
    y = dff.pop('exited')
    X = dff.drop('corporation', axis=1)

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data
    model = model.fit(X, y)

    # write the trained model in a file called trainedmodel.pkl
    with open(model_pth + '/trainedmodel.pkl', 'wb') as file:
        pickle.dump(model, file)
    return None


if __name__ == "__main__":
    # Load config.json and get path variables
    with open('config.json', 'r', encoding='utf8') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])

    train_model(dataset_csv_path, model_path)
