import sys
sys.path.append('../')
from utils import BrainGraphDataset, project_root
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import ConcatDataset
from models import VAE, SameTPred
import copy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    batch_size = 32
    input_dim = 4950

    # define the optimizer and the loss function

    criterion = nn.L1Loss(reduction='sum')

    root = project_root()

    dataroot = 'fc_matrices/psilo_schaefer_before'
    annotations = 'annotations-before.csv'
    before_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')

    dataroot = 'fc_matrices/psilo_schaefer_after'
    annotations = 'annotations-after.csv'
    after_dataset = BrainGraphDataset(img_dir=os.path.join(root, dataroot),
                                annotations_file=os.path.join(root, annotations),
                                transform=None, extra_data=None, setting='upper_triangular')

    dataset = ConcatDataset([before_dataset, after_dataset])

    # Define the train, validation, and test ratios
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # Get the number of samples in the dataset
    num_samples = len(dataset)

    # Calculate the number of samples for each set
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    torch.manual_seed(0)
    # Split the dataset into train, validation, and test sets
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Define the dataloaders for each set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_size, shuffle=False)
        
    num_epochs = 300

    vae = VAE(4950, [128] * 2, 64)
    vae.load_state_dict(torch.load(os.path.join(root, 'vae_weights/vae_dropout_psilo_schaefer_combined_0.pt'), 
                                   map_location=device))
    vae = vae.to(device)

    dropout_list = [0, 0.05 ,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    dropout = dropout_list[int(sys.argv[1]) - 1]
    
    model = SameTPred(64, 256, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Convert the MLP to the device
    model.to(device)

    best_val_loss = float('inf')
    best_state = None

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # train the MLP on the new dataset
    for epoch in range(num_epochs):
        running_loss = 0.0

        model.train()
        vae.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            graphs, labels = data

            graphs = graphs.to(device)
            labels = labels.to(device).float()

            _, _, _, z = vae(graphs.view(-1, input_dim))

            # zero the parameter gradients
            optimizer.zero_grad()

            preds = model(z)

            # calculate the loss and backpropagate
            loss = model.loss(preds, labels.view(preds.shape))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # Validation check
        val_loss = 0.0
        model.eval()
        vae.eval()
        with torch.no_grad():
            for data in val_loader:
                graphs, labels = data

                graphs = graphs.to(device)
                labels = labels.to(device).float()

                _, _, _, z = vae(graphs.view(-1, input_dim))

                preds = model(z)

                val_loss += criterion(preds, labels.view(preds.shape)).item()
        val_loss /= len(val_set)

        # Save the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(vae.state_dict()))
            
    torch.save(best_state[0], os.path.join(root, 'same_t_pred', f'same_t_weight_dropout_{dropout}.pt'))
    torch.save(best_state[1], os.path.join(root, 'same_t_pred', f'vae_unfrozen_dropout_{dropout}.pt'))
        
if __name__ == "__main__":
    main()
    
    