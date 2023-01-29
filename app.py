import json
import os
from flask import Flask, session, jsonify, request
import pandas as pd
# import numpy as np
# import pickle
# import create_prediction_model
from diagnostics import model_predictions, dataframe_summary
from diagnostics import execution_time, missing_data, outdated_packages_list
# import predict_exited_from_saved_model


# Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['prod_deployment_path'])

prediction_model = None


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict() -> str:
    """
    Calls the prediction function on the data in the config file
    under 'output_folder_path'
    """
    filepath = request.args.get('filepath')
    dff = pd.read_csv(filepath)
    preds = model_predictions(dff, model_path)
    return str(preds)

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    """heck the score of the deployed model"""
    with open(dataset_csv_path + '/latestscore.txt', 'r', encoding='utf8'
              ) as f1:
        latestscore = f1.read()
    return latestscore

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary():
    sumamry_dict = dataframe_summary(dataset_csv_path)
    return sumamry_dict


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    """
    Calls the remaingin diagnostics functions and returns the results
    """
    timings = execution_time()
    missing = missing_data(dataset_csv_path)
    outdated = outdated_packages_list()
    dianostics = {'timings': timings, 'missing': missing, 'outdated': outdated}
    return dianostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
