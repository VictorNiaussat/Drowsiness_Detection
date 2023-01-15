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
from dash_player import DashPlayer

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
                                        type='number',
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




def build_tab_2(list_video):
    fig_score = generate_graph_empty()
    return [html.Div(
            [   html.Div([html.Div(
                    [
                        html.Div(
                            [html.H6("ANALYSE MODELE", className="graph__title")]
                        ),
                        html.P(),
                        html.Div(   
                            [       html.Div([html.Label(id="video-select", children="Vidéo :",style={"display":"block","textAlign":"right"})],className="one columns"),
                                    html.Div([dcc.Dropdown(list_video, value=list_video[0], id='video-name-dropdown', clearable=False)],className="two columns"),
                                    html.Div([html.Label(id="model-name-analyse-select", children="Nom :",style={"display":"block","textAlign":"right"})],className="one columns"),
                                    html.Div([dcc.Dropdown(list_model, value=list_model[0], id='modele-name-analyse-dropdown', clearable=False)],className="two columns"),
                                    html.Div([html.Label(id="interval-name-select", children="Intervale :",style={"display":"block","textAlign":"right"})],className="two columns"),
                                    html.Div([dcc.Input(
                                        id="interval-select-input",
                                        type='number',
                                        placeholder="",
                                        style=dict(display='flex', justifyContent='center')
                                    )],className="two columns"),
                                    html.Div([dbc.Button(
                                        "Analyse", id="video-predict-set-btn", n_clicks=0
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
                        html.Div(id='video-loader',
                            children=[html.H6("VIDEO", className="graph__title")]
                        )
                    ],
                    className="two-third column wind__speed__container first",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [   html.Div(),
                                html.P("SCORE ACTUEL", style=dict(display='flex', justifyContent='center')),
                                dcc.Interval(id='video-analyse-interval', interval=1000, n_intervals=-1, disabled=True),
                                daq.LEDDisplay(
                                    id="current-score-led",
                                    value="0.0",
                                    color=list(app_color.values())[1],
                                    backgroundColor=list(app_color.values())[0],
                                    style={"border": 0, 'display':'flex', 'justifyContent':'center'},
                                    size=100,
                                )
                            ],
                            className="graph__container first",
                        )
                    ],
                    className="one-third column histogram__direction",
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
                            [html.H6("EVOLUTION DU SCORE", className="graph__title")]
                        ),
                        dcc.Graph(
                                    id="score-analyse-video", figure=fig_score
                                        )
                    ],
                    className="full column wind__speed__container first",
                )
            ],
            className="app__content first",
        )], className="app__container")]

def build_tab_3():
    pass


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
        dcc.Store(id='df-analyse-video', data=dict())
    ],
)
app.layout = layout
path, list_model = list_model_path()
list_video = load_video_path()
app.validation_layout = html.Div([
    layout,

    build_tab_1(list_model),
    build_tab_2(list_video),
    build_tab_3()
])

@app.callback(
    [Output("app-content-2", "children")],
    [Inp("app-tabs", "value")],
    
)
def render_tab_content(tab_switch):
    if tab_switch == "tab2":
        list_video = load_video_path()
        return build_tab_2(list_video)
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
        State("input-size-select-input", "value"),
        ]
)
def update_graph_analyse_model(n_clicks, model_name, input_size):
    if n_clicks==0:
        raise PreventUpdate
    else:
        path, list_model = list_model_path()
        model_path = os.path.join(path, model_name)
        if not os.path.isfile(os.path.join(model_path, "predictions.csv")):
            make_predictions(model_path, dict(input_size=int(input_size), nb_couches_rentrainement=1, nb_classes=10))
        
        df_predictions = pd.read_csv(os.path.join(model_path, "predictions.csv"))
        
        df_tensorboard = load_tensoboard_data(model_path)
        fig_epoch_accuracy, fig_epoch_loss, fig_confusion_matrix, fig_roc = generate_graph_analyse_model(df_tensorboard, df_predictions)
        return fig_epoch_accuracy, fig_epoch_loss, fig_confusion_matrix, fig_roc



@app.callback(
    Output('current-score-led', 'value'),
    Inp("video-analyse-interval", "n_intervals"),
    [State('df-analyse-video', 'data'),
    State('video-analyse-loc', "currentTime")]
)
def update_video_analyse_current_score(n_intervals, df_score, current_time):
    if df_score == dict():
        return "0.0"
    else:
        df_score = pd.DataFrame.from_dict(df_score)
        current_score = df_score[df_score.t<=int(current_time)].score.iloc[-1]
        current_t = df_score[df_score.t<=int(current_time)].t.iloc[-1]
        next_score = df_score[df_score.t>int(current_time)].score.iloc[0]
        next_t = df_score[df_score.t>int(current_time)].t.iloc[0]
        m = (next_score - current_score)/(next_t-current_t)
        b = current_score - m*current_t
        score =m*current_time + b
        return np.round(score, 2)


@app.callback(
    [Output("score-analyse-video", "figure"),
    Output('video-loader', 'children'),
    Output('df-analyse-video', 'data'),
    Output('video-analyse-interval', 'disabled')],
    Inp("video-predict-set-btn", "n_clicks"),
    [State("modele-name-analyse-dropdown", "value"),
    State("video-name-dropdown", 'value'),
    State("interval-select-input", "value"),]
)
def generate_video_analyse(n_clicks, model_path, video_name, interval):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        video_path_html = os.path.join('/static/',video_name)
        video_path = os.path.join(os.getcwd(),'Data', video_name)
        generator = generator_from_video(video_path, interval)
        specs = dict(input_size=224, nb_couches_rentrainement=4, nb_classes=10) 
        model_path = os.path.join(os.getcwd(), 'Training', 'xception', model_path)
        df_score = make_predictions(model_path, specs, generator)
        fig_score = generate_graph_score(df_score)
        children = [html.Div(
                                [html.H6("VIDEO", className="graph__title")]
                            ),
                                DashPlayer(id='video-analyse-loc',
                                url=video_path_html,
                                        height=720,
                                        width=1280,
                                        controls=True,
                                        style = dict(display='flex', justifyContent='center'))]

        return fig_score, children, df_score.to_dict(), False



if __name__=='__main__':
    app.run_server(debug=True)