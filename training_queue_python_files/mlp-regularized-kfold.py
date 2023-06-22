import sys
sys.path.append('../')

from models import LatentMLP, VAE
from utils import BrainGraphDataset, project_root, get_data_labels
import torch
import torch.optim as optim
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import copy

def main():
    root = project_root()
  
    categories = ['patient_n','condition','bdi_before']

    data_labels = get_data_labels()
    data_labels = data_labels[categories]

    annotations = 'annotations.csv'

    data_labels.loc[data_labels["condition"] == "P", "condition"] = 1
    data_labels.loc[data_labels["condition"] == "E", "condition"] = -1
    data_labels['condition'] = data_labels['condition'].astype('float64')

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataroot = 'fc_matrices/psilo_ica_100_before/'
    psilo_ica_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=data_labels, setting='upper_triangular_and_baseline')
    
    dataroot = 'fc_matrices/psilo_schaefer_before/'
    psilo_schaefer_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=data_labels, setting='upper_triangular_and_baseline')
    
    dataroot = 'fc_matrices/psilo_aal_before/'
    psilo_aal_before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=data_labels, setting='upper_triangular_and_baseline')
    
    configs = [
        (psilo_ica_before_dataset, 'vae_fine_tune_before_dropout_0.pt', 'fine_tune_before'),
        (psilo_ica_before_dataset, 'vae_fine_tune_combined_dropout_0.pt', 'fine_tune_combined'),
        (psilo_ica_before_dataset, 'vae_dropout_psilo_ica_before_0.pt', 'ica_before'),
        (psilo_ica_before_dataset, 'vae_dropout_psilo_ica_combined_0.pt', 'ica_combined'),
        (psilo_schaefer_before_dataset, 'vae_dropout_psilo_schaefer_before_0.pt', 'schaefer_before'),
        (psilo_schaefer_before_dataset, 'vae_dropout_psilo_schaefer_combined_0.pt', 'schaefer_combined'),
        (psilo_aal_before_dataset, 'vae_dropout_psilo_aal_before_0.pt', 'aal_before'),
        (psilo_aal_before_dataset, 'vae_dropout_psilo_aal_combined_0.pt', 'aal_combined'),
    ]
    
    for config in configs:
        # instantiate the VGAE model
        hidden_dim = 64
        latent_dim = 64
        output_dim = 1
        input_dim = 6670 if 'aal' in config[1] else 4950
        lr = 0.001
        batch_size = 8

        vae = VAE(input_dim, [128] * 2, latent_dim)

        # load the trained VGAE weights
        with torch.no_grad():
            vae.load_state_dict(torch.load(os.path.join(root, f'vae_weights/{config[1]}'), map_location=device))

        # define the optimizer and the loss function
        criterion = nn.L1Loss(reduction='sum')

        # Convert the model to the device
        vae.to(device)

        dataset = config[0]

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

        dropout_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        dropout = dropout_list[int(sys.argv[1]) - 1]

        best_perf_label = [0] * num_folds
        best_perf_output = [0] * num_folds
        
        for t, train_loader in enumerate(train_loaders):
            val_loader = val_loaders[t]
            # instantiate the LatentMLP model
            mlp = LatentMLP(latent_dim, hidden_dim, output_dim, dropout=dropout)
            optimizer = optim.Adam(mlp.parameters(), lr=lr)
            # Convert the MLP to the device
            mlp.to(device)

            best_val_loss = float('inf')
            best_mlp_state = None

            # Lists to store training and validation losses
            train_losses = []
            val_losses = []

            # train the MLP on the new dataset
            for epoch in range(num_epochs):
                running_loss = 0.0

                mlp.train()
                vae.train()
                for i, data in enumerate(train_loader, 0):
                    # get the inputs
                    (graphs, base_bdis), labels = data

                    graphs = graphs.to(device)
                    base_bdis = base_bdis.to(device)

                    labels = labels.to(device).float()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # get the latent embeddings from the VGAE
                    with torch.no_grad():
                        _, _, _, zs = vae(graphs.view(-1, input_dim))

                    # pass the latent embeddings through the MLP
                    outputs = mlp(zs, base_bdis)

                    # calculate the loss and backpropagate
                    l1_loss, l2_loss = mlp.loss(outputs, labels.view(outputs.shape))
                    loss = l1_loss + l2_loss
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()

                # Validation check
                val_loss = 0.0

                val_label = []
                val_output = []
                
                mlp.eval()
                vae.eval()
                optimizer.zero_grad()
                with torch.no_grad():
                    for data in val_loader:
                        (graphs, base_bdis), labels = data

                        graphs = graphs.to(device)
                        base_bdis = base_bdis.to(device)

                        labels = labels.to(device).float()

                        # get the latent embeddings from the VGAE
                        _, _, _, zs = vae(graphs.view(-1, input_dim))

                        # pass the latent embeddings through the MLP
                        outputs = mlp(zs, base_bdis)
                        
                        val_label.extend(labels.flatten().tolist())
                        val_output.extend(outputs.flatten().tolist())
                        
                        val_loss += criterion(outputs, labels.view(outputs.shape)).item()
                val_loss /= len(val_set)

                # Save the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_mlp_state = (copy.deepcopy(mlp.state_dict()), copy.deepcopy(vae.state_dict()))
                    
                    best_perf_label[t] = val_label
                    best_perf_output[t] = val_output

                # Print statistics and perform testing every 5 epochs
                if epoch % 30 == 9:
                    print('[%d, %5d] loss: %.3f, val_loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / len(train_loader), val_loss))
                    running_loss = 0.0

                train_losses.append(running_loss / len(train_loader))
                val_losses.append(val_loss)

            torch.save(best_mlp_state[0], os.path.join(root, f'mlp_weights/mlp_weight_dropout_{dropout}_{config[2]}_{t}.pt'))
            torch.save(best_mlp_state[1], os.path.join(root, f'mlp_weights/vae_unfrozen_dropout_{dropout}_{config[2]}_{t}.pt'))
            
            
        results = {'true': best_perf_label, 'pred': best_perf_output}
        import json
        with open(os.path.join(root, f'mlp_weights/true_pred_{dropout}_{config[2]}.json'), 'w') as f:
            json.dump(results, f)
        print('Finished Training')
    
if __name__ == "__main__":
    main()