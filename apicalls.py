"""
Script to call the API endpoints of the model and store the responses
"""
import os
import json
import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

# Call each API endpoint and store the responses
response1 = requests.get(
    URL + "prediction",
    params={'filepath': 'testdata/testdata.csv'},
    timeout=10).json()
response2 = requests.get(URL + "scoring", timeout=10).content.decode()
response3 = requests.get(URL + "summarystats", timeout=10).json()
response4 = requests.get(URL + "diagnostics", timeout=10).json()

apicalls = {'prediction': response1, 'F1_score': response2,
            'data_summary': response3, 'diagnostics': response4}
responses = apicalls

with open(dataset_csv_path + '/apireturns.json', 'w', encoding='utf8') as resp:
    json.dump(responses, resp, indent=4)
