import os
import numpy as np
import boto3
import dash_daq as daq
from dash import dcc, html, Input as Inp, Output, State, ALL, Dash
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
from dash.exceptions import PreventUpdate
import diskcache
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from time import localtime, strftime
from joblib import Memory
import sagemaker
import json
from io import StringIO
from datetime import date, datetime, timedelta
from time import strftime, gmtime
from ast import literal_eval




app = Dash(
    __name__,
    long_callback_manager=lcm,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder ="static",
    assets_url_path="static"
)
app.title = "Dashboard Drowsiness"
application = app.server
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}


#----------------------------S3 Data----------------------------------#

region = "eu-west-1"
prefix = "data"
role = "arn:aws:iam::469563492837:role/service-role/AmazonSageMaker-ExecutionRole-20211008T144923"
bucket_name = 'drowsiness-detection-bucket'
colors_list = px.colors.qualitative.Bold + px.colors.qualitative.Antique


def new_connection(type):
    if type=='all':
        client = boto3.client('s3', region_name=region)
        sm_client = boto3.client('sagemaker', region_name=region)
        session = boto3.Session(region_name=region)
        s3_res = session.resource('s3', region_name=region)
        account = boto3.client("sts", region_name=region).get_caller_identity().get("Account")
        return client, session, sm_client, s3_res, account
    elif type=='sagemaker':
        sm_client = boto3.client('sagemaker', region_name=region)
        return sm_client
    elif type=='s3':
        client = boto3.client('s3', region_name=region)
        session = boto3.Session(region_name=region)
        s3_res = session.resource('s3', region_name=region)
        return client, s3_res
    
    

#------------ Dashboard -----------#


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="ENTRAINEMENT",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="ANALYSE VIDÉO",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Account-analyze-tab",
                        label="VIDÉO TEMPS RÉEL",
                        value="tab3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


