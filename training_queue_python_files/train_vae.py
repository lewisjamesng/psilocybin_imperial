import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
import copy
import os
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import sys
from torch.utils.data import ConcatDataset

sys.path.append('..')
from utils import BrainGraphDataset, project_root
from models import VAE

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    # define the hyperparameters

    hidden_dim = 128
    latent_dim = 64
    lr = 1e-3
    batch_size = 128
    num_epochs = 200

    root = project_root()
    annotations = 'annotations.csv'

    dataroot = 'fc_matrices/hcp_100_ica/'
    hcp_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, dataroot, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')
    
    dataroot = 'fc_matrices/psilo_ica_100_before/'
    psilo_ica_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')
    
    dataroot = 'fc_matrices/psilo_ica_100_after/'
    psilo_ica_after_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')  
    
    dataroot = 'fc_matrices/psilo_schaefer_before/'
    psilo_schaefer_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')
    
    dataroot = 'fc_matrices/psilo_schaefer_after/'
    psilo_schaefer_after_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')  
    
    dataroot = 'fc_matrices/psilo_aal_before/'
    psilo_aal_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')
    
    dataroot = 'fc_matrices/psilo_aal_after/'
    psilo_aal_after_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')  
    
    psilo_ica_combined_dataset = ConcatDataset([psilo_ica_before_dataset, psilo_ica_after_dataset])
    psilo_schaefer_combined_dataset = ConcatDataset([psilo_schaefer_before_dataset, psilo_schaefer_after_dataset])
    psilo_aal_combined_dataset = ConcatDataset([psilo_aal_before_dataset, psilo_aal_after_dataset])
    
    configs = [
        (z, 'hcp'),
        (psilo_ica_before_dataset, 'psilo_ica_before'),
        (psilo_ica_combined_dataset, 'psilo_ica_combined'),
        (psilo_schaefer_before_dataset, 'psilo_schaefer_before'),
        (psilo_schaefer_combined_dataset, 'psilo_schaefer_combined'),
        (psilo_aal_before_dataset, 'psilo_aal_before'),
        (psilo_aal_combined_dataset, 'psilo_aal_combined'),
    ]
    
    for config in configs:
        
        # set the random seed for reproducibility
        torch.manual_seed(0)
        
        dataset = config[0]
        input_dim = 6670 if ('aal' in config[1]) else 4950 # size of the graph adjacency matrix
            
        # split the dataset into training and validation sets
        num_samples = len(dataset)
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # define the data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')  # set to infinity to start
        best_model_state = None

        dropout_list = [0, 0.05 ,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        dropout = dropout_list[int(sys.argv[1]) - 1]

        train_losses = []
        val_losses = []
        model = VAE(input_dim, [hidden_dim] * 2, latent_dim, dropout=dropout).to(device)  # move model to device
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in tqdm(range(num_epochs)):
            train_loss = 0.0
            val_loss = 0.0

            # training
            model.train()
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)  # move data to device
                optimizer.zero_grad()

                recon, mu, logvar, z = model(data.view(-1, input_dim))
                (mse_loss, gmm_loss, l2_reg) = model.loss(recon, data.view(-1, input_dim), mu, logvar, n_components=3)
                loss = mse_loss + gmm_loss
                loss.backward()
                optimizer.step()
                train_loss += mse_loss.item()

            # validation
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(val_loader):
                    data = data.to(device)  # move data to device
                    recon, mu, logvar, z = model(data.view(-1, input_dim))
                    mse_loss, gmm_loss, l2_reg = model.loss(recon, data.view(-1, input_dim), mu, logvar, n_components=3)
                    val_loss += mse_loss.item()
            # append losses to lists
            train_losses.append(train_loss/len(train_dataset))
            val_losses.append(val_loss/len(val_dataset))

            # save the model if the validation loss is at its minimum
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = copy.deepcopy(model.state_dict())

            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}\n')

        # save the best model for this configuration
        torch.save(best_model_state, os.path.join(root, f'vae_weights/vae_dropout_{config[1]}_{dropout}.pt'))
        
if __name__ == "__main__":
    main()