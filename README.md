# Détection de distraction

Le but de ce projet est de réaliser une application qui à partir d'une vidéo, réalise un score de bonne conduite. Le score fait intervenir ces états.

Pour ce travail, on va utiliser la base de donnée d'images venant d'une compétition kaggle :

 https://www.kaggle.com/c/state-farm-distracted-driver-detection 

 Sur cette base de données, nous avons des images de personnes au volant conduisant normalement ou effectuant des actions parallèles. Voici les différentes actions :

- c0 : conduite normale 
- c1 : envoi de SMS - droite 
- c2 : parler au téléphone - droite  
- c3 : envoi de textos - gauche 
- c4 : parler au téléphone - gauche 
- c5 : utiliser la radio 
- c6 : boire 
- c7 : tendre la main derrière soi 
- c8 : coiffer et maquiller 
- c9 : parler au passager

Nous cherchons à réaliser un algorithme de classification sur ces différentes classes.






## Modèle de Deep Learning utilisé

Pour ce projet, nous utiliserons 
