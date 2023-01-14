import os
import numpy as np
import boto3
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io 
import os
import pandas as pd
import numpy as np
import tensorboard as tb
import yaml
import sys
sys.path.append('./')
from SageMaker.Model_Xception.modelClass import model
from sklearn.preprocessing import LabelBinarizer
import plotly.figure_factory as ff
from ast import literal_eval

activity_map = {
    'c0': 'Safe driving', 
    'c1': 'Texting - right', 
    'c2': 'Talking on the phone - right', 
    'c3': 'Texting - left', 
    'c4': 'Talking on the phone - left', 
    'c5': 'Operating the radio', 
    'c6': 'Drinking', 
    'c7': 'Reaching behind', 
    'c8': 'Hair and makeup', 
    'c9': 'Talking to passenger'
        }
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

def new_connection(type, region):
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


def logs_tensorboard_to_dataframe(experiment_id:str):
    """Retourne un DtaFrame avec les données enregistrée par tensorboardpendant l'entrinement du model enregistré au chemin model_path.

    Args:
        experiment_id (str): id de l'experience tensorboard sauvegardée sur tensorboard.dev

    Returns:
        pd.DataFrame: Dataframe([run, tag, value])
    """
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df = df[df.tag.isin(['epoch_accuracy', 'epoch_loss'])]
    df.run = df.run.map(lambda x: x.split('/')[-1])
    return df

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def list_model_path():
    path = os.path.join(os.getcwd(), 'Training', 'xception')
    model_name = get_immediate_subdirectories(path)
    return path, model_name


def load_model(model_path:str, specs:dict): #Utiliser specs rentrée par l'utilisateur plus tard
    """Charge le modèle dont les poids sont stockées dans le dossier au chemin d'accès model_path
        avec les paramètres dans specs.

    Args:
        model_path (str): Chemin d'accès où sont stockés tous les fichiers du modèle.
        specs (dict): Dictionnaire des paramètres du modèle.

    Returns:
        model: Modèle Tensorflow avec les poids chargés.
    """
    m = model(specs['nb_classes'], specs['nb_couches_rentrainement'], specs['input_size'])
    m.load_weights(os.path.join(model_path, 'output', 'model_weights.h5'))
    return m



def make_predictions(model_path:str, specs:dict):
    """Fait les prédictions du modèle dont les poids sont stockées dans le dossier au chemin d'accès model_path
        avec les paramètres specs en utilisant les données de validation de l'entrainement du modèle.

    Args:
        model_path (str): Chemin d'accès où sont stockés tous les fichiers du modèle.
        specs (dict): Dictionnaire des paramètres du modèle.

    Returns:
        path: Chemin d'accès vers le csv contenant les prédictions des images de validation.
    """
    m = load_model(model_path, specs)
    train_datagen = ImageDataGenerator(validation_split=0.2)
    val_generator = train_datagen.flow_from_directory(
            os.path.join('/home/jjean/Documents/GitHub/Drowsiness_Detection/Data/state-farm', 'imgs/train'),
            target_size=(specs['input_size'], specs['input_size']),
            batch_size=1,
            shuffle=False,
            class_mode='categorical',
            subset='validation')
    
    y_pred = m.predict_generator(val_generator, steps=len(val_generator.filenames))
    df = pd.DataFrame(dict(filename=val_generator.filenames, classe=val_generator.classes, prediction=list(y_pred)))
    path = os.path.join(model_path, 'predictions.csv')
    df.to_csv(path)
    return path

def load_tensoboard_data(model_path:str):
    """Renvoie le DataFrame des données tensorboard à partir du chemin d'accès vers les fichiors du modèle.

    Args:
        model_path (str): Chemin d'accès où sont stockés tous les fichiers du modèle.

    Returns:
        pd.DataFrame: DataFrame des données.
    """
    with open(os.path.join(model_path, 'config.yaml')) as f:
        config = yaml.load(f)

    df_tensorboard = logs_tensorboard_to_dataframe(config['tensoboard-dev-id'])
    return df_tensorboard


def generate_graph_analyse_model(df_tensorboard:pd.DataFrame, df_predictions:pd.DataFrame):
    """Génère les graphes d'analyse du modèle à partir des données stockés dans des DataFrames.

    Args:
        df_tensorboard (pd.DataFrame): DataFrame des données Tensorboard.
        df_predictions (pd.DataFrame): DataFrame des données de validation.

    Returns:
        go.Figure: Courbe de l'évolution de l'accuracy au cours des epochs.
        go.Figure: Courbe de l'évolution de la loss au cours des epochs.
        go.Figure: Tableau de la matrice de confusion  des prédictions sur les données de validation.
        go.Figure: Courbe ROC de chaque classe vs les autres classes.
    """
    y_true = df_predictions.classe.to_numpy()
    y_pred = df_predictions[[str(i) for i in range(10)]].to_numpy()
    app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

    colors = [app_color["graph_line"], "#EBF5FA", "#2E5266", "#BD9391"]

    fig_epoch_accuracy = px.line(df_tensorboard[df_tensorboard.tag=='epoch_accuracy'], x='step', y='value', color='run',color_discrete_sequence=colors)
    
    fig_epoch_loss = px.line(df_tensorboard[df_tensorboard.tag=='epoch_loss'], x='step', y='value', color='run',color_discrete_sequence=colors)

    fig_epoch_accuracy.update_layout(plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
                font_color="white",
                legend=dict(title='Run'))
    fig_epoch_accuracy.update_xaxes(title='Epoch', showgrid=False)
    fig_epoch_accuracy.update_yaxes(title='', showgrid=False)

    fig_epoch_loss.update_layout(plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
                font_color="white",
                legend=dict(title='Run'))
    fig_epoch_loss.update_xaxes(title='Epoch', showgrid=False)
    fig_epoch_loss.update_yaxes(title='', showgrid=False)


    layout=dict(
                plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
                font_color="white",
            )
    fig_roc = go.Figure(layout=layout)

    

    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_true = label_binarizer.transform(y_true)

    for i in range(len(activity_map.keys())):
        fpr, tpr, thresholds = roc_curve(y_onehot_true[:, i], y_pred[:, i])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name = f"Class {list(activity_map.values())[i]} vs Rest"
                        ))
    fig_roc.add_shape(
        type='line', line=dict(dash='dash', color=app_color["graph_bg"]),
        x0=0, x1=1, y0=0, y1=1
    )
    fig_roc.update_yaxes(title='Taux de vrais positifs', scaleanchor="x", scaleratio=1,showgrid=False)
    fig_roc.update_xaxes(title = 'Taux de faux positifs',constrain='domain',showgrid=False)

    cm = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
    cm_text = [[str(y) for y in x] for x in cm]
    fig_confusion_matrix =  ff.create_annotated_heatmap(cm, x=list(activity_map.values()), y=list(activity_map.values()), annotation_text=cm_text)
    fig_confusion_matrix.update_layout(layout)

    return fig_epoch_accuracy, fig_epoch_loss, fig_confusion_matrix, fig_roc

def generate_graph_empty():
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