import sys
sys.path.append('../')
from models import VGAE, LatentMLP 
from utils import BrainGraphDataset, project_root, get_data_labels
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = project_root()

    # instantiate the VGAE model
    lr = 0.001
    batch_size = 64

    nf = 1
    ef = 1
    num_nodes = 100
    hidden_dim = 128
    latent_size = 8

    criterion = nn.L1Loss(reduction='sum')
    categories = ['patient_n','condition','bdi_before']

    data_labels = get_data_labels()
    data_labels = data_labels[categories]

    annotations = 'annotations.csv'

    data_labels.loc[data_labels["condition"] == "P", "condition"] = 1
    data_labels.loc[data_labels["condition"] == "E", "condition"] = -1
    data_labels['condition'] = data_labels['condition'].astype('float64')

    dataroot = 'fc_matrices/psilo_schaefer_before/'
    psilo_schaefer_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=data_labels, setting='lz')

    dataroot = 'fc_matrices/psilo_aal_before/'
    psilo_aal_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=data_labels, setting='lz')

    dataroot = 'fc_matrices/psilo_ica_100_before/'
    psilo_ica_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=data_labels, setting='lz')

    configs = [
        (psilo_ica_before_dataset, 'vgae_nf_ica_32_8.pt', 'ica'),
        (psilo_ica_before_dataset, 'vgae_nf_fine_tune.pt', 'fine_tune'),
        (psilo_schaefer_before_dataset, 'vgae_nf_schaefer_32_8.pt', 'schaefer'),
        (psilo_aal_before_dataset, 'vgae_nf_aal_32_8.pt', 'aal'),
    ]

    dropout_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    dropout = dropout_list[int(sys.argv[1]) - 1]
    
    for config in configs:
        dataset = config[0]
        
        torch.manual_seed(0)
        # Split the dataset into train, validation, and test sets

        import torch
        from torch.utils.data import DataLoader
        from sklearn.model_selection import KFold

        # Assuming you have your dataset defined as 'dataset'
        num_folds = 5  # Specify the number of folds
        batch_size = 8  # Specify your desired batch size
        random_seed = 42  # Specify the random seed

        # Create indices for k-fold cross-validation with seeded random number generator
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        # Create empty lists to store train and validation loaders
        train_loaders = []
        val_loaders = []

        for train_index, val_index in kf.split(dataset):
            # Split dataset into train and validation sets for the current fold
            train_set = torch.utils.data.Subset(dataset, train_index)
            val_set = torch.utils.data.Subset(dataset, val_index)

            # Define the dataloaders for the current fold
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            # Append the loaders to the respective lists
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        num_epochs = 500
        import json

        # Dictionary to store training and validation curves
        results = []
        best_set = [0] * num_folds

        for t, train_loader in enumerate(train_loaders):
            val_loader = val_loaders[t]

            num_rois = 116 if 'aal' in config[2] else 100
            vgae = VGAE(1, 1, num_rois, 32, 8, device, dropout=0, l2_strength=0.001).to(device)
            # load the trained VGAE weights
            vgae.load_state_dict(torch.load(os.path.join(root, f'vgae_weights/{config[1]}'), map_location=device))
            # Convert the model to the device
            vgae.to(device)

            best_val_loss = float('inf')  # set to infinity to start
            best_model_state = None
            best_output = None
            train_losses = []
            val_losses = []

            model = LatentMLP(64, 64, 1, dropout=dropout).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)

            src, dest = vgae.edge_index

            for epoch in tqdm(range(num_epochs)):
                train_loss = 0.0
                val_loss = 0.0

                # training
                model.train()
                for batch_idx, ((graph, lz, baseline_bdi), label) in enumerate(train_loader):
                    graph = graph.to(device)  # move data to device
                    lz = lz.to(device)
                    baseline_bdi = baseline_bdi.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()

                    rcn_lz, rcn_edges, z, _, _ = vgae(lz, graph)
                    graph = graph[:, src, dest]

                    output_bdi = model(z.view(z.shape[0], -1), baseline_bdi)

                    l1_loss, l2_loss = model.loss(output_bdi, label.view(output_bdi.shape))
                    loss = l1_loss + l2_loss
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # validation
                model.eval()
                val_label = []
                val_output = []
                val_base = []
                with torch.no_grad():
                    for batch_idx, ((graph, lz, baseline_bdi), label) in enumerate(val_loader):
                        graph = graph.to(device)  # move data to device
                        lz = lz.to(device)
                        baseline_bdi = baseline_bdi.to(device)
                        label = label.to(device)

                        rcn_lz, rcn_edges, z, _, _ = vgae(lz, graph)
                        graph = graph[:, src, dest]

                        output_bdi = model(z.view(z.shape[0], -1), baseline_bdi)

                        val_label.extend(label.flatten().tolist())
                        val_output.extend(output_bdi.flatten().tolist())
                        val_base.extend(baseline_bdi)

                        l1_loss, l2_loss = model.loss(output_bdi, label.view(output_bdi.shape))
                        loss = l1_loss + l2_loss
                        val_loss += loss.item()
                # append losses to lists
                train_losses.append(train_loss/len(train_set))
                val_losses.append(val_loss/len(val_set))

                # save the model if the validation loss is at its minimum
                if val_losses[-1] < best_val_loss:
                    best_val_loss = val_losses[-1]
                    best_set[t] = (val_label, val_output, val_base)
                    best_model_state = (copy.deepcopy(vgae.state_dict()), copy.deepcopy(model.state_dict()))
                # print the losses

    
        total_true = []
        total_pred = []
        total_drug = []
        total_base = []
        
        for i, (true, pred, base) in enumerate(best_set):
            drug = [d[0].item() for d in base]
            base = [d[1].item() for d in base]

            total_true.extend(true)
            total_pred.extend(pred)
            total_drug.extend(drug)
            total_base.extend(base)
            # Calculate R-squared (Pearson correlation coefficient)

        import csv
        # Specify the filename for the CSV file
        filename = os.path.join(root, 'vgae_mlp_results', f'feature-vgae-{config[2]}-{dropout}-results.csv')

        # Create a list of rows with headers
        rows = [['true_post_bdi', 'predicted_post_bdi', 'drug (1 for psilo)', 'base_bdi']]
        for true, pred, drug, base in zip(total_true, total_pred, total_drug, total_base):
            rows.append([true, pred, drug, base])

        # Write the rows to the CSV file
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)


if __name__ == "__main__":
    main()