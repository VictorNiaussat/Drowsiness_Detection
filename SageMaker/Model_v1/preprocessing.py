import numpy as np
from PIL import Image
import pandas as pd
import cv2
from datetime import timedelta


"""distracted_driving_labels = ['Normal Forward Driving',
                              'Drinking',
                              'Phone Call(right)',
                              'Phone Call(left)',
                              'Eating',
                              'Text (Right)',
                              'Text (Left)',
                              'Hair / makeup',
                              'Reaching behind',
                              'Adjust control panel',
                              'Pick up from floor (Driver)',
                              'Pick up from floor (Passenger)',
                              'Talk to passenger at the right',
                              'Talk to passenger at backseat',
                              'Yawning',
                              'Hand on head',
                              'Singing with music',
                              'Shaking or dancing with music']"""

def annotate_video_frames(path_video,path_csv,cameraView = ' Dashboard',appearanceBlock = 'None',freq=30):
  """
  Renvoie un tableau avec les annotations des images correspondantes dans l'ordre.
  Annotations :  voir le tableau distracted driving labels (dans l'ordre 0-17)


  @param path_video: chemin de la vidéo
  @param path_csv: chemin du csv associé à la vidéo
  @param cameraView: String pour savoir quelle partie du CSV lire --> ' Dashboard' ou  ' Rightside_window' ou ' Rear_view'
  Attention à l'espace devant le nom de CameraView ! 
  @param appearanceBlock: String pour savoir quelle partie du CSV lire --> 'None' ou 'Sunglass'
  @param freq: nombre de frames avnt d'extraire une nouvelle image

  @return: (np-array of images, np-array of annotations) 
  ->un tableau avec les annotations des images correspondantes dans l'ordre.
  @raise keyError: si freq<=0
"""
  assert freq>0 , f'Fréquence ne peut pas être nulle ou négative : {freq}'

  labels = pd.read_csv(path_csv)
  cap=cv2.VideoCapture(path_video)
  
  n_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  n_fps = int(cap.get(cv2.CAP_PROP_FPS))
  df = labels[(labels['Camera View']==cameraView)&(labels["Appearance Block"]==appearanceBlock)]

  annotations = []
  images = []
  for frame in range(n_frames):
    if frame%freq==0:
      ret,img=cap.read()
      if ret==False:
        break
      images.append(img)  # Ajout de l'image en cours
      sec = frame//n_fps  # nombre de secondes écoulées à cette frame

      read_time=pd.to_datetime('00:00:00')+timedelta(seconds=sec)  # timeStamp associé à ce nb de sec (date_time 00:00:00 pour être comparable à df)

      current_label = df[ (pd.to_datetime(df["Start Time"])<=read_time) & (pd.to_datetime(df["End Time"])>=read_time) ] # Seule ligne qui matche avec le temps courant
      
      if current_label.empty :
        annotations.append(0)   # pas d'annotations donc conduite non perturbée
      else:
        try : 
          annotations.append(int(current_label["Label/Class ID"])) # Ajoute la classe sous forme d'entier
        except:
          annotations.append(0)  # dans le cas ou le CSV avait un Nan
  return np.array(images),np.array(annotations)
