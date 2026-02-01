# Présentation : 
Projet de deep learning pour la détection non supervisée d'anomalies acoustiques utilisant des Autoencodeurs de Débruitage avec LSTM Bidirectionnel. Le modèle apprend à reconnaître les sons normaux et signale les anomalies via l'erreur de reconstruction.

## Fonctionnalités principales

- Apprentissage non supervisé : Entraîné uniquement sur des sons normaux
- Architecture DAE-BLSTM : Implémentation de l'article de recherche "A NOVEL APPROACH FOR AUTOMATIC ACOUSTIC NOVELTY DETECTION USING A DENOISING AUTOENCODER WITH BIDIRECTIONAL LSTM NEURAL NETWORKS " , 2015
- Dataset ESC-50 : Classification de sons environnementaux (50 catégories) : par souci de taille pour la démo, nous avons choisi ce petit dataset afin de reproduire des sons environnants comme le principe du dataset CHiME utilisé dans l'article de référence.

## Structure du projet
projet/
├── Accoustic Detection.ipynb          # Notebook d'exploration + demo à ouvrir sous Google Collab 
├── models/             # Implémentations DAE-BLSTM et Transformer
    └── blstm_dae.py
    └── transormer.py
├── requirements.txt    # Dépendances
└── README.md

Lien vers le notebook collab : 
https://colab.research.google.com/drive/1QClpG1ewHceEr-CYJcZgmB8-EVMJPDMz?usp=sharing


Lien vers les sources utilisées : 
https://mediatum.ub.tum.de/doc/1253789/file.pdf
https://www.kaggle.com/code/niptloxcompany/mimii-anomaly-transformer-pipeline-pytorch 
https://arxiv.org/abs/1706.03762
