"""
Implementation of the Denoising Autoencoder with Bidirectional LSTM
Based on the paper by Marchi et al. (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAEBLSTM(nn.Module):
    """
    Denoising AE avec BLSTM
    Conforme à l'article de réference 
    
    Différences clés avec autoencodeur standard:
    1. Entraîné sur données corrompues (bruitées)
    2. Doit reconstruire la version PROPRE
    3. Fonction de perte: MSE entre reconstruction et entrée propre
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, noise_std=0.25):
        super().__init__()
        
        self.noise_std = noise_std  # σ = 0.25 comme dans l'article
        self.bidirectional = bidirectional
        
        # === ENCODEUR === 
        self.encoder = nn.LSTM(
            input_size=input_size,           # 54 bandes mel (comme article)
            hidden_size=hidden_size,         # 128-256 comme dans article
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # === DÉCODEUR === 
        # Dans l'article: architecture symétrique
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.decoder = nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # === COUCHE DE RECONSTRUCTION ===
        # Doit retrouver exactement input_size (54)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
            nn.Sigmoid()  # Nos features sont normalisées [0,1]
        )
        
        print(f"[INFO] DAE-BLSTM initialisé avec:")
        print(f"       • Bruit standard: σ={noise_std}")
        print(f"       • Bidirectionnel: {bidirectional}")
        print(f"       • Hidden size: {hidden_size}")
        print(f"       • Couches LSTM: {num_layers}")
    
    def add_noise(self, x):
        """Corruption par bruit Gaussien comme dans l'article"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x
    
    def forward(self, x, return_latent=False):
        """
        Passe forward DAE:
        1. Corruption (en entraînement seulement)
        2. Encodage
        3. Décodage
        4. Reconstruction
        
        
        x: Input propre [batch, seq_len, features]
        return_latent: Si True, retourne aussi l'espace latent
        """
        batch_size, seq_len, n_features = x.shape
        
        # Étape 1: Corruption (DENOISING autoencoder)
        if self.training:
            x_corrupted = self.add_noise(x)
        else:
            x_corrupted = x  # Pas de bruit en inférence
        
        # Étape 2: Encodage
        encoded_outputs, (hidden, cell) = self.encoder(x_corrupted)
        
        # Étape 3: Décodage (à partir du dernier état caché)
        # Initialisation avec les états finaux de l'encodeur
        decoder_input = encoded_outputs  # Dans l'article, ils utilisent la sortie encodée
        
        decoded_outputs, _ = self.decoder(decoder_input)
        
        # Étape 4: Reconstruction des features originales
        reconstructed = self.output_layer(decoded_outputs)
        
        if return_latent:
            return reconstructed, encoded_outputs
        return reconstructed
