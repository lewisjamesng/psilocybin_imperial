import os
import pandas as pd
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import torch
import numpy as np
from numpy import loadtxt
import pickle

def project_root():
    return os.path.dirname(os.path.abspath(__file__))

def make_edge_index(num_nodes):
    return torch.triu(torch.ones((num_nodes, num_nodes)), diagonal=1).nonzero(as_tuple=False).t()

def compress_adjacency_matrix(self, adj_matrix):
    batch_size = adj_matrix.size(0)
    matrix_size = adj_matrix.size(1)
    diag_mask = torch.eye(matrix_size).bool().unsqueeze(0).to(adj_matrix.device)
    mask = ~diag_mask.expand(batch_size, matrix_size, matrix_size)
        
    out = adj_matrix[mask].view(batch_size, matrix_size, matrix_size - 1)
    return out

def triangle(n):
    return n * (n + 1) / 2

def get_data_labels():
    data_labels = f'{project_root()}/data_labels.xlsx'
    df = pd.read_excel(data_labels, nrows=42)
    return df

def graph_to_repr(graph):
    repr = torch.zeros(int(triangle(graph.shape[0] - 1)))
    count = 0
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i > j:
                repr[count] = graph[i][j]
                count += 1
    return repr

class BrainGraphDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, extra_data=None, linear=True, labelled=False, setting='normal', class_weight=None):
        
        self.class_weight = class_weight
        
        with open(annotations_file) as f:
            num_columns = len(f.readline().split(','))

        # Read the CSV file based on the number of columns
        if num_columns == 3:
            self.graph_labels = pd.DataFrame(pd.read_csv(annotations_file, usecols=[0, 1, 2], names=['filename', 'patient_n', 'final_integration_bdi'], header=None))
        else:
            self.graph_labels = pd.DataFrame(pd.read_csv(annotations_file, usecols=[0, 1], names=['filename', 'final_integration_bdi'], header=None))
            
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.extra_data = extra_data
        self.linear = linear
        self.setting = setting

    def __len__(self):
        return len(self.graph_labels)
    
    def add_data_labels(self, repr, idx):
        tocat = self.get_data_label(idx)
        repr_end = torch.cat([repr, tocat])
        return repr_end
    
    def get_data_label(self, idx):
        patient_n = self.graph_labels.iloc[idx, 1]
        if self.extra_data is None:
            return torch.zeros((2)).float()
        match = self.extra_data.loc[self.extra_data['patient_n'] == patient_n]
        return torch.tensor(match.drop(['patient_n'], axis=1).values).view(-1).float()
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.graph_labels.iloc[idx, 0]))
        
        graph = np.loadtxt(img_path, delimiter=',', dtype=np.float64)
        
        if self.setting == 'upper_triangular_baseline_lz':
            lz_path = os.path.join(project_root(), 'lempel_ziv', self.img_dir.split("/")[-1], str(self.graph_labels.iloc[idx, 0]))
            lz = np.genfromtxt(lz_path, delimiter=',')
            repr_end = (graph_to_repr(graph), self.get_data_label(idx).float(), torch.tensor(lz).float().view(lz.shape[0], -1))
            
        elif self.setting == 'upper_triangular':
            repr_end = graph_to_repr(graph)
        
        elif self.setting == 'class_weighted_graph':
            repr_end = (torch.tensor(graph).float(), self.class_weight)
        
        elif self.setting == 'upper_triangular_and_baseline':
            repr_end = (graph_to_repr(graph).float(), self.get_data_label(idx).float())
            
        elif self.setting == 'graph_and_baseline':
            repr_end = (torch.tensor(graph).float(), self.get_data_label(idx).float())

        elif self.setting == 'lz':
            lz_path = os.path.join(project_root(), 'lempel_ziv', self.img_dir.split("/")[-2], str(self.graph_labels.iloc[idx, 0]))
            lz = np.genfromtxt(lz_path, delimiter=',')
            repr_end = (torch.tensor(graph).float(), torch.tensor(lz).float().view(lz.shape[0], -1), self.get_data_label(idx).float())
        
        elif self.setting == 'graph':
            repr_end = torch.tensor(graph).float()
        
        label = self.graph_labels.iloc[idx, -1]
        
        return repr_end, label
