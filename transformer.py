# ============================================
# ANOMALY TRANSFORMER PIPELINE (PyTorch)
# ============================================
# https://www.kaggle.com/code/niptloxcompany/mimii-anomaly-transformer-pipeline-pytorch 

import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class TransformerDAE(nn.Module):
    def __init__(
        self,
        input_dim=64,
        d_model=128,
        num_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        noise_std=0.1,
        max_len=500
    ):
        super().__init__()

        self.noise_std = noise_std

        # Projection d'entrée
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Encodage de position appris
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Reconstruction
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        B, T, _ = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x + self.pos_embedding[:, :T, :]

        x = self.encoder(x)
        return self.output_proj(x)
