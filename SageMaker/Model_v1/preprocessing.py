import numpy as np
import pandas as pd
import cv2
from datetime import timedelta


"""distracted_driving_labels = ['Normal Forward Driving', 0 
                              'Drinking', 6 
                              'Phone Call(right)', 2 
                              'Phone Call(left)', 4 
                              'Eating', 6 
                              'Text (Right)', 1 
                              'Text (Left)', 3
                              'Hair / makeup',  8 
                              'Reaching behind', 7 
                              'Adjust control panel', 5 
                              'Pick up from floor (Driver)', 5 
                              'Pick up from floor (Passenger)', 5 
                              'Talk to passenger at the right', 9 
                              'Talk to passenger at backseat', 9 
                              'Yawning', 0 
                              'Hand on head',   0 
                              'Singing with music', 0 
                              'Shaking or dancing with music' 0 ]"""

"""
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
"""



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

  @return: (images (np-array) ,annotations (np-array of int) )
  ->un tableau avec les annotations des images correspondantes dans l'ordre.
  @raise keyError: si freq<=0
"""
  assert freq>0 , f'Fréquence ne peut pas être nulle ou négative : {freq}'

  distracted_driving_to_map = {
    0 : 0,
    1 : 6,
    2 : 2,
    3 : 4,
    4 : 6,
    5 : 1,
    6 : 3,
    7 : 8,
    8 : 7,
    9 : 5,
    10:5,
    11:5,
    12:9,
    13:9,
    14:0,
    15:0,
    16:0,
    17:0
  }

  labels = pd.read_csv(path_csv)
  cap=cv2.VideoCapture(path_video)
  
  n_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  n_fps = int(cap.get(cv2.CAP_PROP_FPS))
  df = labels[(labels['Camera View']==cameraView)&(labels["Appearance Block"]==appearanceBlock)]

  annotations = []
  images = []
  for frame in range(n_frames):
    if frame%freq==0:
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
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
          annotations.append(distracted_driving_to_map[int(current_label["Label/Class ID"])]) # Ajoute la classe sous forme d'entier
        except:
          annotations.append(0)  # dans le cas ou le CSV avait un Nan
  return np.array(images),np.array(annotations)
