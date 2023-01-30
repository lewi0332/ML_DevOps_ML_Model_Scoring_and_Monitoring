# ML_DevOps_ML_Model_Scoring_and_Monitoring
Project 4 for Udacity Machine Learning DevOps Engineer Nanodegree



```
import dash
app = dash.Dash(__name__)
server = app.server

@server.route('/hello')
def hello():
    return 'Hello, World!'

```