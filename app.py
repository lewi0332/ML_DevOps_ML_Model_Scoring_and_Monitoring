"""
Script to run the Flask app

Author: Derrick Lewis
Date: 2023-01-29
"""
import json
import os
from flask import Flask, request
import pandas as pd
# import create_prediction_model
from diagnostics import model_predictions, dataframe_summary
from diagnostics import execution_time, missing_data, outdated_packages_list


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['prod_deployment_path'])

prediction_model = None


@app.route("/prediction", methods=['GET', 'OPTIONS'])
def predict() -> str:
    """
    Calls the prediction function on the data in the config file
    under 'output_folder_path'
    """
    filepath = request.args.get('filepath')
    dff = pd.read_csv(filepath)
    preds = model_predictions(dff, model_path)
    return str(list(preds))


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """Check the score of the deployed model"""
    with open(dataset_csv_path + '/latestscore.txt', 'r', encoding='utf8'
              ) as f1:
        latestscore = f1.read()
    return latestscore


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary():
    """
    Calls the summary function on the data in the config file
    """
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
    dianostics = {'timings': timings, 'missing_data': missing, 'outdated_pckgs': outdated}
    return dianostics


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
