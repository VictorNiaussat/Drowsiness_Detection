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

#------------------------------Config------------------------------#

mem = Memory(location='/tmp/joblib')
pd.options.plotting.backend = "plotly"
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
cache = diskcache.Cache('./cache')
lcm = DiskcacheLongCallbackManager(cache)

app = Dash(
    __name__,
    long_callback_manager=lcm,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder ="static",
    assets_url_path="static"
)
app.title = "DBSCAN Dashboard"
application = app.server
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

#----------------------------S3 Data----------------------------------#

region = "eu-west-1"
prefix = "sagemaker/DBSCAN"
role = "arn:aws:iam::469563492837:role/service-role/AmazonSageMaker-ExecutionRole-20211008T144923"
bucket_name = 'anti-bot-detection'
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

def get_models_datas(client):
    response = client.list_objects_v2(Bucket='anti-bot-detection', Prefix='models')
    files = response.get('Contents')
    files = [file['Key'].replace('/output/model.tar.gz','').replace('models/step-function-training-ablstm-v2-','') for file in files  if "step-function-training-ablstm-v2" in file['Key']]
    files.append('Modèle en utilisation')
    response_data = client.list_objects_v2(Bucket='anti-bot-detection', Prefix='inference/results/real-time/')
    datas = response_data.get('Contents')
    datas = [data['Key'].replace('inference/results/real-time/','').replace('.csv','') for data in datas if '.csv' in data['Key']]
    return files, datas

client,_,sm_client,_,_ = new_connection('all')
files, datas = get_models_datas(client)   

def infos_model(date, sm_client):
    if date== "Modèle en utilisation":
        infos_endpoint = sm_client.describe_endpoint(EndpointName='anti-bot-detection-test')
        variant_name = infos_endpoint['ProductionVariants'][0]['VariantName']
        endpoint_config_name = infos_endpoint['EndpointConfigName']
        infos_endpoint_config = sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        model_name = str()
        for variant in infos_endpoint_config['ProductionVariants']:
            if variant['VariantName'] == variant_name:
                model_name = variant['ModelName']
        infos_model = sm_client.describe_model(ModelName=model_name)
        model_path = infos_model['PrimaryContainer']['ModelDataUrl']
        date_training = model_path.replace("s3://anti-bot-detection/models/step-function-training-ablstm-v2-","").replace("/output/model.tar.gz", "")
        environment = infos_model['PrimaryContainer']['Environment']
        return model_path, environment, date_training
    else:
        infos_model = sm_client.describe_model(ModelName=f"model-inference-{date}")
        model_path = infos_model['PrimaryContainer']['ModelDataUrl']
        environment = infos_model['PrimaryContainer']['Environment']
        return model_path, environment, date

def init_model(date, sm_client):
    _, environment, date_training = infos_model(date, sm_client)
    dict_events = json.loads(client.get_object(Bucket=bucket_name, Key=f"training/dict/dico_events_{date_training}.json")['Body'].read().decode('utf-8'))
    vocab_size = len(dict_events)+1
    state_dict = dict(vocab_size=vocab_size, embedding_dims=environment['embedding_dims'], rnn_units=environment['rnn_units'],max_seq=environment['max_seq'], eps=5e-3, min_sample=100, perplexity=30)
    return state_dict


state_dict = init_model(files[-1], sm_client)


def make_dbscan(session, account, model_path, data_path, vocab_size,embedding_dims, rnn_units, max_seq, eps, min_sample, perplexity, model_in_use):
    sess = sagemaker.Session(boto_session=session)
    output_prefix = f"dashboard-dbscan/resultat_{data_path.split('/')[-1]}"
    dbscan = sagemaker.estimator.Estimator(
        "{}.dkr.ecr.{}.amazonaws.com/dbscan:latest".format(account, region),
        role,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        output_path="s3://{}/{}/dbscan/output".format(bucket_name, prefix),
        input_mode= "File",
        sagemaker_session=sess,
        hyperparameters=dict(
            eps=eps,
            min_pts=min_sample,
            dim=2,
            embedding_dims=embedding_dims,
            rnn_units=rnn_units,
            max_seq=max_seq,
            vocab_size=vocab_size,
            perplexity=perplexity
        ),
        environment=dict(model_in_use=model_in_use, model_path=model_path, prefix = output_prefix)
    )
    dbscan.fit(inputs=data_path)
   

def generate_empty_graph():
    layout = {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "No data Available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ],
            "plot_bgcolor":app_color["graph_bg"],
            "paper_bgcolor":app_color["graph_bg"],
            "font_color":"white"
        }
    fig = go.Figure(layout=layout)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig


def generate_graph(df_fig):
    app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
    if len(df_fig)>0:
        df_fig = df_fig.sort_values(by='cluster')
        df_fig['cluster'] = df_fig['cluster'].astype('str')
        fig = px.scatter(df_fig,x="x",y="y",color="cluster", color_discrete_sequence=colors_list)
        fig.update_xaxes(showgrid=False,
                                    showline=False,
                                    showticklabels=False,
                                    zeroline=False,
                                    title='',
                                    range=[int(df_fig['x'].min())-1,int(df_fig['x'].max())+1])
        fig.update_yaxes(showgrid=False,
                                    showline=False,
                                    showticklabels=False,
                                    zeroline=False,
                                    title='',
                                    range=[int(df_fig['y'].min())-1,int(df_fig['y'].max())+1])

        fig.update_layout(plot_bgcolor=app_color["graph_bg"],
                    paper_bgcolor=app_color["graph_bg"],
                    font_color="white",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=760)
    else:
        fig = px.scatter(df_fig,x="x",y="y",color="cluster", color_discrete_sequence=colors_list)
        fig.update_xaxes(showgrid=False,
                                    showline=False,
                                    showticklabels=False,
                                    zeroline=False,
                                    title='',)
        fig.update_yaxes(showgrid=False,
                                    showline=False,
                                    showticklabels=False,
                                    zeroline=False,
                                    title='',)

        fig.update_layout(plot_bgcolor=app_color["graph_bg"],
                    paper_bgcolor=app_color["graph_bg"],
                    font_color="white",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=760)
    return fig


ud_threshold_input = daq.NumericInput(
    id="ud_threshold_input", className="setting-input", size=150,min=0,  max=1,value=0.8
)
ud_eps_input = daq.NumericInput(
    id="ud_eps_input", className="setting-input", size=150,min=1e-7,  max=1,value=state_dict['eps']
)
ud_perplexity_input = daq.NumericInput(
    id="ud_perplexity_input", className="setting-input", size=150,min=20,  max=200,value=state_dict['perplexity']
)
ud_min_sample_input = daq.NumericInput(
    id="ud_min_sample_input", className="setting-input", size=150,min=1,  max=1000,value=state_dict['min_sample']
)
layout_specs = [
    ud_threshold_input,
    ud_eps_input,
    ud_perplexity_input,
    ud_min_sample_input
]

def build_banner():
    return html.Div([
                html.Div(
                    [
                        html.H4("Dashboard Analyse DBScan", className="app__header__title"),
                        html.P(
                            "Ce dashboard permet l'analyse des données récoltées dans l'optique de trouver de nouveaux types de bots et de trouver des correlations entre eux",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src=app.get_asset_url("ankama-crop.png"),
                                className="app__menu__img",
                            ),
                            href="https://www.ankama.com/fr",
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
                        id="Specs-tab",
                        label="PARAMÈTRES",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="ANALYSE",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Account-analyze-tab",
                        label="ACCOUNT",
                        value="tab3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )


def build_tab_1(files, datas):
    return [# Manually select metrics
        html.Div(
            id="set-specs-intro-container",
            className='twelve columns',
            children=html.P(
                "Paramétrage du modèle et des données pour analyse."
            ),
        ),
        html.Div(
            id="settings-menu",
            children=[
                html.Div(
                    id="metric-select-menu",
                    className='three columns',
                    children=[
                        html.Label(id="metric-select-title", children="Séléctionner date d'entrainement"),
                        html.Br(),
                        dcc.Dropdown(
                            id="metric-select-dropdown",
                            options=list(
                                {"label": file, "value": file} for file in files
                            ),
                            value=files[-1],
                        ),
                        html.Br(),
                        html.Label(id="data-select-title", children="Séléctionner données de test"),
                        html.Br(),
                        dcc.Dropdown(
                            id="data-select-dropdown",
                            options=list(
                                {"label": data, "value": data} for data in datas
                            ),
                            multi=True
                        ),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            id="button-div",
                            children=[html.Div(id='button-and-loading',className='twelve columns',
                            children = [
                                html.Div(children=[html.Button("Analyse", id="value-setter-set-btn")], className='six columns'),
                                html.Div(children=[html.Progress(id="progress_bar")], className='six columns')
                                        ])
                                
                            ],
                        ),
                    ],
                )]),
                html.Div(
                    id="value-setter-menu",
                    className='nine columns',
                    children=[],
                )
                ]


layout_tab_1 = html.Div(build_tab_1(files, datas))

def build_value_setter_line(line_num, label, value, col3):
    return html.Div(
        id=line_num,
        children=[
            html.Label(label, className="four columns"),
            html.Label(value, className="four columns"),
            html.Div(col3, className="four columns"),
        ],
        className="row",
    )


def build_setter_menu(state_dict):
    children = [
            html.Div(id="value-setter-panel", children=[
                build_value_setter_line(
                "value-setter-panel-header",
                "Paramètres",
                "Données initiales",
                "Nouvelle valeur",
            ),
            build_value_setter_line(
                "value-setter-panel-usl",
                "Dimension de l'embedding",
                state_dict["embedding_dims"],
                "Non modifiable",
            ),
            build_value_setter_line(
                "value-setter-panel-lsl",
                "Dimension des cellules RNN",
                state_dict["rnn_units"],
                "Non modifiable",
            ),
            build_value_setter_line(
                "value-setter-panel-ucl",
                "Longueur de la game session",
                state_dict["max_seq"],
                "Non modifiable",
            ),
            build_value_setter_line(
                "value-setter-panel-lcl",
                "Taille du vocabulaire",
                state_dict["vocab_size"],
                "Non modifiable",
            ),
            build_value_setter_line(
                "value-setter-panel-lcl",
                "EPS pour DBSCAN",
                state_dict["eps"],
                ud_eps_input,
            ),
            build_value_setter_line(
                "value-setter-panel-lcl",
                "Min Sample pour DBSCAN",
                state_dict["min_sample"],
                ud_min_sample_input,
            ),
            build_value_setter_line(
                "value-setter-panel-lcl",
                "Perplexité pour T-SNE",
                state_dict["perplexity"],
                ud_perplexity_input,
            ),
            ])
        ]
    return children


def generate_metric_row(id, style, col1, col2, col3, col4):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"margin-right": "2.5rem", "minWidth": "50px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center"},
                className="one column",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%", "verticalAlign": "center"},
                className="six columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={"display": "flex", "justifyContent": "center"},
                className="one column",
                children=col4["children"],
            ),
        ],
    )

def generate_account_row(id, style, col1, col2, col3, col4, col5):
    if style is None:
        style = {"height": "12rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one columns",
                style={"margin-right": "2.5rem", "minWidth": "50px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center"},
                className="one columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%"},
                className="seven columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={"justifyContent": "center"},
                className="one columns",
                children=col4["children"],
            ),
            html.Div(
                id=col5["id"],
                style={"justifyContent": "center"},
                className="one columns",
                children=col5["children"],
            ),
        ],
    )

def generate_cluster_row(id, style, col1, col2, col3, col4):
    if style is None:
        style = {"height": "3rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row cluster-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="two columns",
                style={"textAlign": "center", "margin-right": "2.5rem", "minWidth": "50px"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center"},
                className="two columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"textAlign": "center"},
                className="one columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={"textAlign": "center", "justifyContent": "left"},
                className="height columns",
                children=col4["children"],
            ),
        ],
    )

def generate_metric_list_header():
    return generate_metric_row(
        "metric_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("CLUSTER")},
        {"id": "m_header_4", "children": html.Div("POP")},
        {"id": "m_header_5", "children": html.Div("RÉPARTITION")},
        {"id": "m_header_6", "children": html.Div("MOYENNE PRÉDICTION")},
    )

def generate_account_list_header():
    return generate_account_row(
        "account_header",
        {"height": "3rem", "textAlign": "center"},
        {"id": "a_header_1", "children": html.Div("ACCOUNT ID")},
        {"id": "a_header_2", "children": html.Div("SESSION ID")},
        {"id": "a_header_3", "children": html.Div("GAME SESSION")},
        {"id": "a_header_4", "children": html.Div("DURÉE")},
        {"id": "a_header_5", "children": html.Div("PRÉDICTION")},
    )

def generate_cluster_list_header():
    return generate_cluster_row(
        "cluster_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("ACCOUNT_ID")},
        {"id": "m_header_4", "children": html.Div("SESSION_ID")},
        {"id": "m_header_5", "children": html.Div("PRÉDICTION")},
        {"id": "m_header_6", "children": html.Div("CLUSTER")},
    )

def generate_metric_list_content(df,cluster):
    df2 = df[df.cluster==cluster]
    taille_totale = len(df)
    fig_box = px.box(df[df.cluster==cluster], x='predictions', notched=True,range_x=[0,1])
    fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        #height=150,
                        #width=700,
                        margin=dict(l=50, r=50, t=0, b=0))
    fig_box.update_traces(marker=dict(color=app_color["graph_line"]))
    fig_box.update_xaxes(title='',showline=False,
                                showgrid=False,
                                zeroline=False)
    return generate_metric_row(
        f"metric_content_{cluster}",
        {"height": "10rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": f"m_content_1_{cluster}", "children": str(cluster)},
        {"id": f"m_content_2_{cluster}", "children": f"{np.round(100*len(df2)/taille_totale,1)}%"},
        {"id": f"m_content_3_{cluster}", "children": dcc.Graph(id=f'jsp_{cluster}', figure = fig_box,config={'displayModeBar': False},style={"width": "100%", "height": "95%"})},
        {"id": f"m_content_4_{cluster}", "children": str(df2.predictions.mean())[:4]}
    )

def generate_gantt_figure(events, dates, levels, df_taggage, dict_events):
    last_date = datetime.fromtimestamp(dates[0]-10).strftime('%Y-%m-%d %H:%M:%S')
    task = ["Evenement" for _ in range(len(events))]
    colors = px.colors.qualitative.Bold + px.colors.qualitative.Antique
    evenement = [f"Evenement n° {i} : {df_taggage[df_taggage['Event ID']==dict_events[events[i-1]]].Name.values[0]}" if dict_events[events[i-1]] < 1000 else f"Evenement n° {i} : sort provenant du cluster n° {dict_events[events[i-1]]}" for i in range(1,len(events)+1)]
    discrete_color_map = {evenement[i-1]: colors[(events[i-1]-1) % len(colors)] for i in range(1,len(events)+1)}
    start = []
    finish = []
    for i,d in enumerate(dates):
        start.append(last_date)
        finish.append(datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))
        if i<len(dates)-1:
            if dates[i+1]!=d:
                last_date=datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S')
    df_gantt = pd.DataFrame(dict(Task=task, Début=start, Fin=finish, Evenement=evenement, events=events, Level=levels))
    df_gantt = df_gantt.sort_values(["Fin", "events"], ascending = True)
    scale = [0]
    for i in range(1,len(df_gantt)):
        if df_gantt['Fin'].iloc[i] == df_gantt['Fin'].iloc[i-1]:
            scale.append(scale[i-1]+1)
        else:
            scale.append(0)
    df_gantt['Scale']= scale
    fig = px.timeline(df_gantt, x_start="Début", x_end="Fin", y="Task", color="Evenement",hover_name="Evenement", hover_data =dict(Task=False, Début=True, Fin=True, Level=True, Evenement=False), color_discrete_map=discrete_color_map)

    fig.update_xaxes(showgrid=False,
                                showline=False,
                                showticklabels=False,
                                zeroline=False,
                                title='')
    fig.update_yaxes(showgrid=False,
                                showline=False,
                                showticklabels=False,
                                zeroline=False,
                                title='')
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False)
    scales = df_gantt.Scale.tolist()
    for i,s in enumerate(scales):
        if s>0:
            max=1
            j=i
            while j<len(scales) and scales[j]!=0:
                max=scales[j]
                j+=1
            max += 1 
            fig.data[i].width = (max-s)*0.8/max
            fig.data[i].offset = -0.4
    return fig

def generate_account_list_content(df, sm_client):
    _, environment,_ = infos_model("Modèle en utilisation", sm_client)
    dict_events = {int(v): int(k) for k, v in json.loads(
    client.get_object(Bucket='anti-bot-detection',
                      Key=f"training/dict/{environment['dict_events']}"
                      )['Body'].read().decode('utf-8')).items()}
    df_taggage = pd.read_csv("s3://anti-bot-detection/training/dict/taggage_DOFUS.csv", sep=';')
    children=[]
    for i in range(len(df)):
        id = f"{df.account_id.iloc[i]}_{df.session_id.iloc[i]}"
        events = literal_eval(df.events.iloc[i])
        dates = literal_eval(df.dates.iloc[i])
        levels = literal_eval(df.character_levels.iloc[i])
        ts = dates[-1]-dates[0]
        children.append(generate_account_row(
            f"account_content_{id}",
            {"height": "10rem", "textAlign": "center"},
            {"id": f"a_content_1_{id}", "children": df.account_id.iloc[i]},
            {"id": f"a_content_2_{id}", "children": df.session_id.iloc[i]},
            {"id": f"a_content_3_{id}", "children": dcc.Graph(id=f'visu_{id}', figure = generate_gantt_figure(events, dates, levels, df_taggage, dict_events),config={'displayModeBar': False},style={"width": "100%", "height": "100%"})},
            {"id": f"a_content_4_{id}", "children": str(timedelta(seconds=ts))},
            {"id": f"a_content_5_{id}", "children": str(df.prediction.iloc[i])[:4]}
        ))
    return [html.Div(children)]

def generate_cluster_description(df, cluster):
    if cluster==-1:
        return "Cette session de jeu fait partie du cluster -1, il ne s'agit pas réellement d'un cluster mais plutôt des sessions de jeu qui n'ont pas pu être mise dans un clsuter, difficile d'en tirer une conclusion..."
    else:
        infos = df[df.cluster==cluster].predictions.describe()
        mean = infos['mean']
        std = infos['std']
        if mean>=0.91 and std<0.05:
            return f"Cette session de jeu fait partie du cluster {cluster}, il s'agit d'un cluster de bots avec très peu de chance pour qu'il n'en soit pas un."
        elif mean>=0.91 and std>0.05:
            return f"Cette session de jeu fait partie du cluster {cluster}, il s'agit d'un cluster de bots mais qui contient des sessions de jeu pouvant appartenir à celle d'un humain."
        elif mean>=0.4:
            return f"Cette session de jeu fait partie du cluster {cluster}, il s'agit d'un cluster difficilement analysable, à voir avec l'oeil humain"
        else:
            return f"Cette session de jeu fait partie du cluster {cluster}, il s'agit d'un cluster d'humains, rian à signaler pour cette session de jeu."


def generate_cluster_list_content(df):
    children = []
    for i in range(len(df)):
        id = f"{df.account_id.iloc[i]}_{df.session_id.iloc[i]}"
        cluster = df.cluster.iloc[i]
        cluster_description = generate_cluster_description(df, cluster)
        children.append(generate_cluster_row(
            f"cluster_content_{id}",
            {"height": "10rem", "margin": "1rem 0"},
            {"id": f"c_content_1_{id}", "children": df.account_id.iloc[i]},
            {"id": f"c_content_2_{id}", "children": df.session_id.iloc[i]},
            {"id": f"c_content_3_{id}", "children": str(df.predictions.iloc[i])[:4]},
            {"id": f"c_content_4_{id}", "children": cluster_description}
        ))
    return [children]

def generate_cluster_list(clusters):
    children = []

    for cluster in clusters:
        children.append(html.Div(id=f"row-{cluster}", className = "row metric-row", style = {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
                        children = [html.Div(id=f'check-{cluster}',children=[dcc.Checklist([""], id={"type":"compile-cluster-check", "index":int(cluster)})], className="two columns", style={"textAlign": "center"}),
                                    html.Div(id=f'name-{cluster}',children=[f"Cluster {cluster}"], className="two columns"), 
                                    html.Div(id=f'button-{cluster}',children=[dbc.Button("EXPORT",id={"type":"export-cluster-btn", "index":int(cluster)},
                                                                                 style={"color":"rgb(255, 255, 255)", "border":"1px solid #007ACE", 
                                                                                            "width":"100%", "height":"95%"}, n_clicks=0)],
                                                                                 className="three columns",
                                                                                 )
                                    
                        ]))
    return children

def build_tab_2(df_fig): 
    fig = generate_graph(df_fig)
    row_content = []
    clusters = list(df_fig.cluster.unique())
    clusters.sort()
    for cluster in clusters:
        row_content.append(generate_metric_list_content(df_fig, cluster))
        
    return [html.Div(
    [
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("VISUALISATION CLUSTERS", className="graph__title")]
                        ),
                        dcc.Graph(
                            id="precision", figure=fig,config={
        'displayModeBar': False
    }
                                )
                    ],
                    className="full column wind__speed__container first",
                )
            ],
            className="app__content first",
        ),
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("ANALYSE CLUSTERS", className="graph__title")]
                        ),
                        html.P(),
                        html.Div(
                            id="metric-summary-session",
                            className="twelve columns",
                            children=[
                                html.Div(
                                    id="metric-div",
                                    children=[
                                        generate_metric_list_header(),
                                        html.Div(
                                            id="metric-rows",
                                            children=row_content
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                    className="two-third-extand column wind__speed__container first",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            'EXPORT',
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(id="metric_header-2", children=[html.Div(id="export-cluster-header", className='row metric-row', style = {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
                children = [html.Div("FUSIONNER", className= "two columns"),
                html.Div("CLUSTER", className = "two columns"),
                html.Div("EXPORTER", className="three columns")
                ])]),
                                html.Div(id="metric-rows-2", children=generate_cluster_list(clusters)),
                                html.Div(id='export-cluster-combine',className="right-btn-display",
                                        children=[
                                            
                                                dbc.Button("EXPORT FUSION", id='set-export-cluster-combine-btn')
                                            ])
                            ],
                            className="graph__container",
                        )
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content first",
        ),
        html.Div([
            html.Div(children = [
                html.Div([
                    html.H6("ANALYSE ACCOUNT_ID PAR RAPPORT AUX CLUSTERS", className="graph__title"),
                    html.Div(id="gs-clust-analyse-content",
                    children = [
                        html.Div([html.Label(id="data-select-account", children="Noms de compte :",style={"display":"block","textAlign":"right","vertical-align": "middle"})],style={"padding-top":"1rem"}, className="two columns"),
                        html.Div([dcc.Input(
                            id="account-cluster-select-input",
                            type='text',
                            placeholder="Exemple : 1111111,2222222",
                            style=dict(display='flex', justifyContent='center')
                        )],className="two columns"),
                        html.Div([dbc.Button(
                                    "ANALYSE", id="accounts-gs-clust-set-btn", n_clicks=0
                                )],className="one columns")
                            ,
                    
                    ],style=dict(paddingTop="1rem")),
                    html.Div([
                    html.Div(id="analyse-gs-clust-header", children=[generate_cluster_list_header()]),
                    html.Div(id="result-analyse-gs-clust",
                        children = [])])
                    
            ], className = "full column wind__speed__container first")
        ], 
        className = "app__content fisrt")])
    ],
    className="app__container",
)]

layout_tab_2 = build_tab_2(pd.DataFrame(columns=["account_id", "session_id", "x", "y", "z", "prediction", "cluster"]))

def build_tab_3():
    return [html.Div(
            [   html.Div([html.Div(
                    [
                        html.Div(
                            [html.H6("ANALYSE CLUSTERS", className="graph__title")]
                        ),
                        html.P(),
                        html.Div(
                            [       html.Div([html.Label(id="data-select-title", children="Noms de compte :",style={"display":"block","textAlign":"right","vertical-align": "middle"})],className="two columns"),
                                    html.Div([dcc.Input(
                                        id="account-select-input",
                                        type='text',
                                        placeholder="Exemple : 1111111,2222222",
                                        style=dict(display='flex', justifyContent='center')
                                    )],className="three columns"),
                                    html.Div([html.Label(id="data-select", children="Dates :",style={"display":"block","textAlign":"right"})],className="one columns"),
                                    html.Div([dcc.DatePickerRange(
                                        id='account-date',
                                        min_date_allowed=date(1995, 8, 5),
                                        max_date_allowed=date.today(),
                                        minimum_nights=0,
                                        initial_visible_month=date.today(),
                                        end_date=date.today(),
                                        style=dict(display='flex', justifyContent='center'),
                                        display_format='DD/MM/YYYY',
                                        with_portal=True
                                    )], className="three columns"),
                                    html.Div([dbc.Button(
                                        "Affichage", id="accounts-gs-set-btn", n_clicks=0
                                    )],className="one columns"),
                                    html.Div([], className="four columns")
                            ], className="twelve columns"
                        ),
                        html.P()
                    ],
                    className="full column wind__speed__container first",
                )],className='app__content first'),
                html.Div( [html.Div(children=[
                    html.Div(
                            id="account-summary-session",
                            className="twelve columns",
                            children=[
                                html.Div(
                                    id="account-div",
                                    children=[
                                        html.P(),
                                        generate_account_list_header(),
                                        html.Div(
                                            id="account-rows",
                                            children=[]
                                        ),
                                    ],
                                ),
                            ],
                        )
                ], className="full column wind__speed__container second")
            ],
            className="app__content first",
        )], className="app__container")]

layout_tab_3 = build_tab_3()
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
        dcc.Store(id='specs',data=state_dict),
        dcc.Download(id="export-cluster"),
        dcc.Download(id="export-cluster-combine-download"),
        dcc.Store(id='data_name', data={"data_name":""})
    ],
)
app.layout = layout

app.validation_layout = html.Div([
    layout,
    layout_specs,
    layout_tab_1,
    layout_tab_2,
    layout_tab_3
])

@app.callback(
    output=Output("app-content-2", "children"),
    inputs=[Inp("app-tabs", "value")],
    state=[State('data_name', 'data')]  
)
def render_tab_content(tab_switch, data_name):
    if tab_switch == "tab2":
        if data_name["data_name"]=="":
            df = pd.DataFrame(columns=["account_id", "session_id", "x", "y", "z", "prediction", "cluster"])
        else:
            data_name = data_name['data_name']
            df = pd.read_csv(f's3://anti-bot-detection/dashboard-dbscan/resultat_{data_name}')
        return build_tab_2(df)
    elif tab_switch == "tab1":
        client, _ = new_connection('s3')
        files, datas = get_models_datas(client)   
        return build_tab_1(files, datas)
    return build_tab_3()


@app.callback(
    output=[Output('specs', 'data'),
            Output("value-setter-menu", "children")],
    inputs=[Inp("metric-select-dropdown", "value")],
)
def update_specs(date):
    sm_client = new_connection('sagemaker')
    _, environment, date_training = infos_model(date, sm_client)
    dict_events = json.loads(client.get_object(Bucket=bucket_name, Key=f"training/dict/dico_events_{date_training}.json")['Body'].read().decode('utf-8'))
    vocab_size = len(dict_events)+1
    new_state_dict = dict(vocab_size=vocab_size, embedding_dims=environment['embedding_dims'], rnn_units=environment['rnn_units'],max_seq=environment['max_seq'], eps=state_dict["eps"], min_sample=state_dict["min_sample"], perplexity=state_dict["perplexity"])
    children = build_setter_menu(new_state_dict)
    return state_dict, children


@app.long_callback(
    output=Output("data_name", "data"),
    inputs=(
        Inp("value-setter-set-btn", "n_clicks"),
        State("metric-select-dropdown", "value"),
        State("data-select-dropdown", "value"),
        State("specs", "data"),
        State("ud_eps_input", "value"),
        State("ud_min_sample_input", "value"),
        State("ud_perplexity_input", "value")),
    running=[(Output("value-setter-set-btn", "disabled"), True, False),
            (Output("Control-chart-tab", "disabled"), True, False),
            (Output("Account-analyze-tab", "disabled"), True, False)],
    progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    prevent_initial_call=True
)
def analyse(set_progress, set_btn,model, data, state_dict, eps, min_sample, perplexity):
    set_progress((0,100))
    _, session, sm_client, s3_res, account = new_connection("all")
    if model=='Modèle en utilisation':
        model_in_use="True"
        model_path, _, _ = infos_model(model, sm_client)
        model_path  = model_path.replace('s3://anti-bot-detection/','')
    else:
        model_in_use="False"
        model_path = f'models/step-function-training-ablstm-v2-{model}/output/model.tar.gz'
    set_progress((10,100))
    if len(data)==0:
        raise PreventUpdate
    elif len(data)==1:
        data_name = f"{data[0]}.csv"
        make_dbscan(session, account, model_path, f"s3://anti-bot-detection/inference/results/real-time/{data_name}", state_dict["vocab_size"], state_dict["embedding_dims"], state_dict["rnn_units"], state_dict["max_seq"], eps, min_sample, perplexity, model_in_use)
        
    else:
        df = pd.read_csv(f"s3://anti-bot-detection/inference/results/real-time/{data[0]}.csv", index_col=False)
        for d in data[1:]:
            df = pd.concat([df, pd.read_csv(f"s3://anti-bot-detection/inference/results/real-time/{d}.csv", index_col=False)])
        data_name = f"to_dbscan_{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}.csv"
        s3_object_name = f"dashboard-dbscan/{data_name}"
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_res.Object(bucket_name, s3_object_name).put(Body=csv_buffer.getvalue())
        del df
        make_dbscan(session, account, model_path, f"s3://anti-bot-detection/dashboard-dbscan/{data_name}", state_dict["vocab_size"], state_dict["embedding_dims"], state_dict["rnn_units"], state_dict["max_seq"], eps, min_sample, perplexity, model_in_use)
        s3_res.Object('anti-bot-detection', s3_object_name).delete()
    set_progress((100,100))
    return {"data_name": data_name}


@app.callback(
    [Output("export-cluster", "data"),
    Output({'type': 'export-cluster-btn', 'index': ALL},'n_clicks')],
    Inp({'type': 'export-cluster-btn', 'index': ALL}, 'n_clicks'),
    [State('data_name', 'data')]
)
def export_data(n_clicks_list, data_name):
    for i,n_clicks in enumerate(n_clicks_list):
        cluster = i-1
        if n_clicks >0:
            data_name = data_name['data_name']
            df = pd.read_csv(f's3://anti-bot-detection/dashboard-dbscan/resultat_{data_name}')
            return dcc.send_data_frame(df[df.cluster==cluster][['account_id','session_id']].to_csv, f"gs_cluster_{cluster}_{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.csv", index=False), [0 for _ in range((len(n_clicks_list)))]
    raise PreventUpdate


@app.callback(
    [Output("export-cluster-combine-download", "data"),
    Output({'type': 'compile-cluster-check', 'index': ALL}, 'value')],
    Inp("set-export-cluster-combine-btn", 'n_clicks'),
    [State({'type': 'compile-cluster-check', 'index': ALL}, 'value'),
    State('data_name', 'data')]
)
def export_combine_data(n_clicks, liste_checkbox, data_name):
    if n_clicks is None:
        raise PreventUpdate
    else:
        idx_clust = []
        clusters = ""
        for i,check in enumerate(liste_checkbox):
            if check:
                idx_clust.append(i-1)
                clusters += f"{i-1}_"
        if len(idx_clust)==0:
            raise PreventUpdate
        else:
            data_name = data_name['data_name']
            df = pd.read_csv(f's3://anti-bot-detection/dashboard-dbscan/resultat_{data_name}')
            return dcc.send_data_frame(df[df.cluster.isin(idx_clust)][['account_id','session_id']].to_csv, f"gs_clusters_{clusters[:-1]}_{strftime('%Y-%m-%d-%H-%M-%S', localtime())}.csv", index=False), [[] for _ in range(len(liste_checkbox))]


@app.callback(
    output=[Output("account-rows", "children")],
    inputs=[
        Inp("accounts-gs-set-btn", "n_clicks")
    ],
    state=[State("account-select-input", "value"),
           State("account-date", "start_date"),
           State("account-date", "end_date")])
def generate_gs_visualisation(n_clicks, accounts_id, date_debut, date_fin):
    if n_clicks==0:
        raise PreventUpdate
    else:
        sm_client= new_connection('sagemaker')
        list_accounts = [int(account_id) for account_id in accounts_id.split(',')]
        date_debut_ = date.fromisoformat(date_debut)
        date_fin_ = date.fromisoformat(date_fin)
        datas_visualisation = [data for data in datas if (datetime.strptime(data.replace("predictions_",""), "%Y_%m_%d").date()>=date_debut_) and (datetime.strptime(data.replace("predictions_",""), "%Y_%m_%d").date()<=date_fin_)]
        df = pd.read_csv(f"s3://anti-bot-detection/inference/results/real-time/{datas_visualisation[0]}.csv", index_col=False)
        for d in datas_visualisation[1:]:
            df = pd.concat([df, pd.read_csv(f"s3://anti-bot-detection/inference/results/real-time/{d}.csv", index_col=False)])
        df = df[df.account_id.isin(list_accounts)]
        df = df.sort_values(by=['account_id', 'prediction'])
        return generate_account_list_content(df, sm_client)



@app.callback(
    output=[Output("result-analyse-gs-clust", "children")],
    inputs=[
        Inp("accounts-gs-clust-set-btn", "n_clicks")
    ],
    state=[State("account-cluster-select-input", "value"),
           State('data_name', 'data')])
def generate_cluster_description_visualisation(n_clicks, accounts_id, data_name):
    if n_clicks==0:
        raise PreventUpdate
    else:
        list_accounts = [int(account_id) for account_id in accounts_id.split(',')]
        if data_name["data_name"]=="":
            df = pd.DataFrame(columns=["account_id", "session_id", "x", "y", "z", "prediction", "cluster"])
        else:
            data_name = data_name['data_name']
            df = pd.read_csv(f's3://anti-bot-detection/dashboard-dbscan/resultat_{data_name}')
        df = df[df.account_id.isin(list_accounts)]
        return generate_cluster_list_content(df)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port='8090')