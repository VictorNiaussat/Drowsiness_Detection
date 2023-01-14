import os
import numpy as np
import boto3
import dash_daq as daq
from dash import dcc, html, Input as Inp, Output, State, ALL, Dash
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
from dash.exceptions import PreventUpdate
import diskcache
import plotly.express as px
import pandas as pd
from utils import *

app = Dash(
    __name__,
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


    

#------------ Dashboard -----------#

def build_banner():
    return html.Div([
                html.Div(
                    [
                        html.H4("Dashboard Drowsiness Detection", className="app__header__title"),
                        html.P(
                            "Ce dashboard permet lbla-bla-bla", #A modifier
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src=app.get_asset_url("centrale-logo.png"),
                                className="app__menu__img",
                            ),
                            href="https://github.com/VictorNiaussat/Drowsiness_Detection",
                        ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        )

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
                        id="analyse-model-tab",
                        label="ENTRAINEMENT",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="analyse-video-tab",
                        label="ANALYSE VIDÉO",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="real-time-video-tab",
                        label="VIDÉO TEMPS RÉEL",
                        value="tab3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def build_tab_1(list_model):
    fig_epoch_accuracy, fig_epoch_loss, fig_confusion_matrix, fig_roc = generate_graph_empty(), generate_graph_empty(), generate_graph_empty(), generate_graph_empty()

    return [html.Div(
            [   html.Div([html.Div(
                    [
                        html.Div(
                            [html.H6("ANALYSE MODELE", className="graph__title")]
                        ),
                        html.P(),
                        html.Div(   
                            [       html.Div([html.Label(id="model-select", children="Modèle :",style={"display":"block","textAlign":"right"})],className="one columns"),
                                    html.Div([dcc.Dropdown(options={'Xception': "Modèle Xception"}, value='Xception', id='modele-dropdown', clearable=False)],className="two columns"),
                                    html.Div([html.Label(id="model-name-select", children="Nom :",style={"display":"block","textAlign":"right"})],className="one columns"),
                                    html.Div([dcc.Dropdown(list_model, value=list_model[0], id='modele-name-dropdown', clearable=False)],className="two columns"),
                                    html.Div([html.Label(id="input-size-name-select", children="Input Size :",style={"display":"block","textAlign":"right"})],className="two columns"),
                                    html.Div([dcc.Input(
                                        id="input-size-select-input",
                                        type='number    ',
                                        placeholder="",
                                        style=dict(display='flex', justifyContent='center')
                                    )],className="two columns"),
                                    html.Div([dbc.Button(
                                        "Analyse", id="accounts-gs-set-btn", n_clicks=0
                                    )],className="one columns"),
                                    html.Div([], className="four columns")
                            ], className="twelve columns"
                        ),
                        html.P()
                    ],
                    className="full column wind__speed__container first",
                )],className='app__content first'),
                ## A partir de là
                
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("ACCURACY", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="accuracy", figure=fig_epoch_accuracy
                                )
                    ],
                    className="one-half column wind__speed__container first",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            'LOSS',
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="loss", figure=fig_epoch_loss
                                        )
                            ],
                            className="graph__container first",
                        )
                    ],
                    className="one-half column histogram__direction",
                ),
            ],
            className="app__content first",
        ),
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("MATRICE DE CONFUSION", className="graph__title")]
                        ),
                        dcc.Graph(
                                    id="confusion-matrix", figure=fig_confusion_matrix
                                        )
                    ],
                    className="one-half column wind__speed__container first",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            'Courbe ROC',
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="roc", figure=fig_roc
                                        ),
                            ],
                            className="graph__container",
                        )
                    ],
                    className="one-half column histogram__direction",
                ),
            ],
            className="app__content first",
        )], className="app__container")]


def build_tab_2():
    pass

def build_tab_3():
    pass



app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        print(os.getcwd()),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content-2"),
            ],
        ),
        dcc.Store(id='weights',data=str()),
    ],
)

layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content-2"),
            ],
        ),
        dcc.Store(id='specs',data=dict()),
    ],
)
app.layout = layout
path, list_model = list_model_path()

app.validation_layout = html.Div([
    layout,

    build_tab_1(list_model),
    build_tab_2(),
    build_tab_3()
])

@app.callback(
    [Output("app-content-2", "children")],
    [Inp("app-tabs", "value")],
    
)
def render_tab_content(tab_switch):
    if tab_switch == "tab2":
        return build_tab_2()
    elif tab_switch == "tab3":
        return build_tab_3()
    return build_tab_1(list_model=list_model)


@app.callback(
    [Output('accuracy', 'figure'),
            Output('loss', 'figure'),
            Output('confusion-matrix', 'figure'),
            Output('roc', 'figure')],
    Inp("accounts-gs-set-btn", "n_clicks"),
    [
        State("modele-name-dropdown", "value"),
        State("input-size-name-select", "value"),
        ]
)
def update_graph_analyse_model(n_clicks, model_name, input_size):
    if n_clicks==0:
        raise PreventUpdate
    else:
        path, list_model = list_model_path()
        model_path = os.path.join(path, model_name)
        """print(os.path.join(model_path, "predictions.csv"))
        if not os.path.isfile(os.path.join(model_path, "predictions.csv")):
            make_predictions(model_path, dict(input_size=int(input_size), nb_couches_rentrainement=1, nb_classes=10))
        """
        
        df_predictions = pd.read_csv(os.path.join(model_path, "predictions.csv"))
        
        df_tensorboard = load_tensoboard_data(model_path)
        fig_epoch_accuracy, fig_epoch_loss, fig_confusion_matrix, fig_roc = generate_graph_analyse_model(df_tensorboard, df_predictions)
        return fig_epoch_accuracy, fig_epoch_loss, fig_confusion_matrix, fig_roc


if __name__=='__main__':
    app.run_server()