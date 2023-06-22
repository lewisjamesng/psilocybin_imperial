from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
import random
from utils import make_edge_index

class SameTPred(nn.Module):
    def __init__(self, input_dim, hidden_dim: int, dropout=0.1):
        super(SameTPred, self).__init__()
        self.reg = nn.Dropout(p=dropout)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.lr = nn.LeakyReLU()
        self.criterion = nn.L1Loss(reduction='sum')
        
    def forward(self, graph):
        out = self.fc1(graph)
        out = self.lr(out)
        out = self.reg(out)
        
        out = self.fc2(out)
        out = self.lr(out)
        out = self.reg(out)
        
        out = self.fc3(out)
        out = self.lr(out)
        out = self.reg(out)
    
        out = self.fc4(out)
        return out
    
    def loss(self, ground, pred):
        l1_loss = self.criterion(ground, pred)
        l2_loss = 0.
        
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
            
        return l1_loss + 0.01 * l2_loss

class LatentMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout=0.1, extra_dim=0):
        super(LatentMLP, self).__init__()
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.reg = nn.Dropout(p=dropout)
        
        self.fc1 = nn.Linear(latent_dim + 2 + extra_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.lr = nn.LeakyReLU()
        self.criterion = nn.L1Loss(reduction='sum')
        
    def forward(self, z, baseline_bdi):
        out = torch.cat([z, baseline_bdi], dim=1)

        out = self.fc1(out)
        out = self.lr(out)
        out = self.reg(out)
        
        out = self.fc2(out)
        out = self.lr(out)
        out = self.reg(out)
        
        out = self.fc3(out)
        out = self.lr(out)
        out = self.reg(out)
    
        out = self.fc4(out)
        return out
    
    def loss(self, ground, pred):
        l1_loss = self.criterion(ground, pred)
        l2_loss = 0.
        
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
            
        return l1_loss, 0.01 * l2_loss

class Encoder(nn.Module):
    def __init__(self, nf, ef, num_nodes, hidden_dim, latent_size, device=torch.device('cpu'), dropout=0.):
        super(Encoder, self).__init__()
        self.device = device
        self.nf = nf
        self.num_nodes = num_nodes
        
        self.dropout = nn.Dropout(dropout)
        self.lr = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Linear(1 + nf + ef * num_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_size)
        
        self.fc5 = nn.Linear(latent_size * num_nodes, 256)
        self.fc6 = nn.Linear(256, 128)

        
    def forward(self, x, graph):
        node_num = torch.arange(self.num_nodes).view(self.num_nodes, self.nf).to(self.device)
        node_num = node_num.repeat(x.shape[0], 1, 1)

        out = torch.cat([node_num, x, graph], dim=2)

        out = self.dropout(self.lr(self.fc1(out)))
        out = self.dropout(self.lr(self.fc2(out)))  
        out = self.dropout(self.lr(self.fc3(out)))  
        out = self.dropout(self.lr(self.fc4(out)))  
        
        out = out.view(out.shape[0], -1)
        
        out = self.dropout(self.lr(self.fc5(out)))
        out = self.fc6(out)

        return torch.chunk(out, 2, dim=1)
    
    
# class EncoderV1(nn.Module):
#     def __init__(self, nf, ef, num_nodes, hidden_dim, latent_size, device=torch.device('cpu'), dropout=0.):
#         super(Encoder, self).__init__()
#         self.device = device
#         self.nf = nf
#         self.num_nodes = num_nodes
        
#         self.dropout = nn.Dropout(dropout)
#         self.lr = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()
        
#         self.fc1 = nn.Linear(1 + nf + ef * num_nodes, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, latent_size)
        
#         self.fc3 = nn.Linear(latent_size * num_nodes, 256)
#         self.fc4 = nn.Linear(256, 128)

        
#     def forward(self, x, graph):
#         node_num = torch.arange(self.num_nodes).view(self.num_nodes, self.nf).to(self.device)
#         node_num = node_num.repeat(x.shape[0], 1, 1)

#         out = torch.cat([node_num, x, graph], dim=2)

#         out = self.dropout(self.lr(self.fc1(out)))
#         out = self.dropout(self.lr(self.fc2(out)))  
        
#         out = out.view(out.shape[0], -1)
        
#         out = self.dropout(self.lr(self.fc3(out)))
#         out = self.fc4(out)

#         return torch.chunk(out, 2, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim, latent_size, dropout=0.):
        super(Decoder, self).__init__()
        
        self.decoder_nodes = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout), 
            nn.Linear(64, num_nodes)
        )
        
        self.decoder_edges = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout), 
            nn.Linear(256, int((num_nodes - 1) * num_nodes / 2))
        )
        
    def forward(self, z):
        rcn_x = self.decoder_nodes(z)
        rcn_edges = self.decoder_edges(z)
        return rcn_x, rcn_edges
    
    
class EncoderNoNF(nn.Module):
    def __init__(self, ef, num_nodes, hidden_dim, latent_size, device=torch.device('cpu'), dropout=0.):
        super(EncoderNoNF, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        
        self.dropout = nn.Dropout(dropout)
        self.lr = nn.LeakyReLU()
        
        self.fc1 = nn.Linear(1 + ef * num_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_size)
        
        self.fc5 = nn.Linear(latent_size * num_nodes, 256)
        self.fc6 = nn.Linear(256, 128)

    def forward(self, graph):
        node_num = torch.arange(self.num_nodes).view(-1, 1).to(self.device)
        node_num = node_num.repeat(graph.shape[0], 1, 1)

        out = torch.cat([node_num, graph], dim=2)

        out = self.dropout(self.lr(self.fc1(out)))
        out = self.dropout(self.lr(self.fc2(out)))
        out = self.dropout(self.lr(self.fc3(out)))
        out = self.dropout(self.lr(self.fc4(out)))
        
        out = out.view(out.shape[0], -1)
        
        out = self.dropout(self.lr(self.fc5(out)))
        out = self.fc6(out)

        return torch.chunk(out, 2, dim=1)
    
class DecoderNoNF(nn.Module):
    def __init__(self, num_nodes, hidden_dim, latent_size, dropout=0.):
        super(DecoderNoNF, self).__init__()
        
        self.decoder_edges = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout), 
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout), 
            nn.Linear(128, int((num_nodes - 1) * num_nodes / 2))
        )
        
    def forward(self, z):
        rcn_edges = self.decoder_edges(z)
        return rcn_edges
    
class Discriminator(torch.nn.Module):
    def __init__(self, latent_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(nn.Linear(68, 32), nn.LeakyReLU(), nn.Linear(32, 1))
        
    def forward(self, h, s):
        return F.sigmoid(self.disc(torch.cat([h, s])))      
    
class VGAE(nn.Module):
    def __init__(self, nf, ef, num_nodes, hidden_dim, latent_size, device, dropout=0., l2_strength=0.001, im_strength=1.0, use_nf=True):
        super(VGAE, self).__init__()
        
        self.edge_index = make_edge_index(num_nodes) 

        self.l2_strength = l2_strength
        self.im_strength = im_strength
        self.use_nf = use_nf
        
        if use_nf:
            self.encoder = Encoder(nf, ef, num_nodes, hidden_dim, latent_size, device, dropout)
            self.decoder = Decoder(num_nodes, 256, latent_size, dropout)
        else:
            self.encoder = EncoderNoNF(ef, num_nodes, hidden_dim, latent_size, device, dropout)
            self.decoder = DecoderNoNF(num_nodes, 256, latent_size, dropout)
            
#         self.disc = Discriminator(64)
#         self.readout = nn.Sequential(
#             nn.Linear(400, 64), 
#             nn.LeakyReLU(), 
#             nn.Linear(64, 64),
#             nn.LeakyReLU(), 
#             nn.Linear(64, 64),
#             nn.Sigmoid(),
#         )
        
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x, edges):
        if self.use_nf:
            mu, logvar = self.encoder(x, edges)
            z = self.reparameterize(mu, logvar)
            rcn_x, rcn_edges = self.decoder(z)
            return rcn_x, rcn_edges, z, mu, logvar
        else:
            mu, logvar = self.encoder(edges)
            z = self.reparameterize(mu, logvar)
            rcn_edges = self.decoder(z)
            return rcn_edges, z, mu, logvar

    def loss(self, x, rcn_x, edges, rcn_edges, mu, logvar):
        MSE_edges = nn.functional.mse_loss(rcn_edges, edges, reduction='sum')
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
    #         im_loss = self.infomax_loss(z) * self.im_strength
        l2_loss = self.calc_l2_loss() * self.l2_strength
        if self.use_nf:
            MSE_x = nn.functional.mse_loss(rcn_x, x, reduction='sum')
            return (MSE_x, MSE_edges, kl_loss, l2_loss)
        return (MSE_edges, kl_loss, l2_loss)
        
    def calc_l2_loss(self):
        # L2 regularization
        l2_loss = 0.
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                l2_loss += torch.norm(param, p=2)

        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                l2_loss += torch.norm(param, p=2)
        return l2_loss
    
#     def readout(self, hs):
#         return torch.cat([torch.mean(hs, dim=1), torch.std(hs, dim=1)], dim=1)
    
#     def infomax_loss(self, hs, sample_prop=0.5):
#         im_loss = 0.
#         ss = self.readout(hs.view(hs.shape[0], -1))
#         n = len(hs) 
#         n_samples = int(n * sample_prop)
#         for i in range(n):
#             n1 = i
#             n2 = (n1 + 1) % n
            
#             # each representation gets compared to an adjacent one
#             hs_0 = hs[n1]
#             hs_1 = hs[n2]
#             s_0 = ss[n1]
#             s_1 = ss[n2]
        
#             l0 = 0.
#             l1 = 0.
#             for j in random.sample(range(n), n_samples):
#                 l0 += torch.log(self.disc(hs_0[j], s_0))

#             for j in random.sample(range(n), n_samples):
#                 l1 += torch.log(1 - self.disc(hs_1[j], s_0))

#             obj_loss = l0 + l1

#             im_loss += obj_loss / (hs_0.shape[0] + hs_1.shape[0])
            
#         return -im_loss



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, dropout=0., l2_strength=0.001):
        super(VAE, self).__init__()

        self.l2_strength = l2_strength
        # Input layer
        self.input_layer_enc = nn.Linear(input_dim, hidden_sizes[0])
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Hidden layers
        self.hidden_layers_enc = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers_enc.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # Output layer
        self.output_layer_enc = nn.Linear(hidden_sizes[-1], latent_dim * 2)
        
        # Input layer
        self.input_layer_dec = nn.Linear(latent_dim, hidden_sizes[-1])
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Hidden layers
        self.hidden_layers_dec = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1)[::-1]:
            self.hidden_layers_dec.append(nn.Linear(hidden_sizes[i+1], hidden_sizes[i]))

        # Output layer
        self.output_layer_dec = nn.Linear(hidden_sizes[0], input_dim)
        
        
    def encode(self, x):
        x = self.input_layer_enc(x)
        x = self.relu(x)
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers_enc:
            x = hidden_layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.output_layer_enc(x)
        
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def decode(self, z):
        x = self.input_layer_dec(z)
        x = self.relu(x)
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers_dec:
            x = hidden_layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.output_layer_dec(x)
        return x
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        return x_recon, mu, logvar, z

    def loss(self, x_recon, x, mu, logvar, n_components=3, class_weights=None):
        if class_weights is None:
            MSE = nn.functional.mse_loss(x_recon, x, reduction='sum')

            # Calculate the GMM loss
            gmm = GaussianMixture(n_components=n_components)
            z = mu.detach().cpu().numpy()
            gmm.fit(z)
            gmm_loss = -gmm.score(z)
        else:
            MSE = torch.sum(nn.functional.mse_loss(x_recon, x, reduction='none'),dim=1) @ class_weights
            
            # Calculate the GMM loss
            gmm = GaussianMixture(n_components=n_components)
            z = mu.detach().cpu().numpy()
            gmm.fit(z)
            
            if self.device is None:
                gmm_loss = -torch.mean(torch.tensor(gmm.score_samples(z)).float() * class_weights)
            else:
                gmm_loss = -torch.mean(torch.tensor(gmm.score_samples(z)).to(self.device).float() * class_weights)

        # L2 regularization
        l2_reg = 0.
        for param in self.parameters():
            l2_reg += torch.linalg.norm(param)

        return (MSE, gmm_loss, self.l2_strength * l2_reg)


