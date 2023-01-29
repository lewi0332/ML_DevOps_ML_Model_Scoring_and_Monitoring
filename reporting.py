"""
This script is used to generate a confusion matrix using the test data
and the deployed model. The confusion matrix is saved to the workspace

Author: Derrick Lewis
Date: 2023-01-29
"""
import json
import os
import logging
import pandas as pd
from sklearn import metrics
import plotly.graph_objects as go
from diagnostics import model_predictions

logging.basicConfig(
    filename="./logs/diagnostics.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def score_model(
    dff: pd.DataFrame,
    output_folder_path: str,
    prod_deployment_path: str
        ) -> None:
    """calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    Parameters
    ---
    dff: pd.DataFrame
        A dataframe containing the test data
    output_folder_path: str
        Path to the directory containing the output folder
    prod_deployment_path: str
        Path to the directory containing the deployed model

    Returns
    ---
    fig: plotly.graph_objects.Figure
        A plotly figure containing the confusion matrix

    """
    logging.info('Generating confusion matrix')
    try:
        preds = model_predictions(
            dff, prod_deployment_path)
    except Exception as e:
        logging.error('Error loading model predictions: %s', e)
        raise e
    logging.info('loaded model predictions')

    logging.info('Generating confusion matrix in Plotly')
    print('dff exited type: ', dff['exited'].dtype)
    print('preds type: ', preds.dtype)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=metrics.confusion_matrix(dff['exited'].astype(int), preds),
        x=['Predicted Not Exited', 'Predicted Exited'],
        y=['Actual Not Exited', 'Actual Exited'],
        text=metrics.confusion_matrix(dff['exited'].astype(int), preds),
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='YlGnBu')
        )
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
        )

    logging.info('Saving confusion matrix to %s', output_folder_path)
    fig.write_image(
        os.path.join(output_folder_path, 'confusionmatrix.png'),
        format='png',
        width=800, height=600)

    # TODO: write json format for dashboard
    logging.info('Finished generating confusion matrix')
    return fig


if __name__ == '__main__':

    # Load config.json and get path variables
    with open('config.json', 'r', encoding="utf8") as f:
        config = json.load(f)

    OUTPUT = os.path.join(config['output_folder_path'])
    MODEL = os.path.join(config['prod_deployment_path'])
    TEST_DATA = os.path.join(config['test_data_path'])
    DFF = pd.read_csv(TEST_DATA + '/testdata.csv')

    FIG = score_model(DFF, OUTPUT, MODEL)
    FIG.show()
