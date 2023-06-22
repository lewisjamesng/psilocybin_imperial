import torch.nn as nn
import torch
import os
import sys
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import copy

sys.path.append('../')
from utils import BrainGraphDataset, get_data_labels, project_root
from old_models import SimpleFCNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # set the random seed for reproducibility
    torch.manual_seed(0)

    # define the hyperparameters
    input_dim = 4950
    hidden_dim = 128
    dropout = 0.2

    lr = 1e-3
    batch_size = 8
    num_epochs = 200

    annotations = 'annotations.csv'

    parent_dir = project_root()

    dataroot = 'fc_matrices/psilo_schaefer_before/'

    categories = ['patient_n', 'condition', 'bdi_before']

    data_labels = get_data_labels()
    data_labels = data_labels[categories]

    data_labels.loc[data_labels["condition"] == "P", "condition"] = 1
    data_labels.loc[data_labels["condition"] == "E", "condition"] = -1
    data_labels['condition'] = data_labels['condition'].astype('float64')

    dataset = BrainGraphDataset(img_dir=os.path.join(parent_dir, dataroot),
                                annotations_file=os.path.join(parent_dir, annotations),
                                transform=None, extra_data=data_labels, setting='upper_triangular_and_baseline')

    # split the dataset into training and validation sets
    test_index = int(sys.argv[1]) - 1
    test_indices = [test_index]
    non_test_indices = [i for i in range(len(dataset)) if i != test_index]

    test_set = torch.utils.data.Subset(dataset, test_indices)
    dataset = torch.utils.data.Subset(dataset, non_test_indices)

    # Define the train, validation, and test ratios
    train_ratio = 0.8
    val_ratio = 0.2

    # Get the number of samples in the dataset
    num_samples = len(dataset)

    # Calculate the number of samples for each set
    train_size = int(train_ratio * num_samples)
    val_size = num_samples - train_size

    # Split the dataset into train, validation, and test sets
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define the dataloaders for each set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_indices), shuffle=False)

    # define a dictionary to store the loss curves for each configuration
    loss_curves = {}

    best_val_loss = float('inf')  # set to infinity to start
    best_model_state = None

    train_losses = []
    val_losses = []
    hidden = [hidden_dim] * 4
    model = SimpleFCNN(input_dim, hidden, dropout=dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0.0
        val_loss = 0.0

        # training
        model.train()
        for batch_idx, ((fc, base), label) in enumerate(train_loader):
            fc = fc.to(device)  # move data to device
            base = base.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(fc, base)

            (mae_loss, l2_reg) = model.loss(output, label.view(output.shape))
            loss = mae_loss + l2_reg
            loss.backward()
            optimizer.step()
            train_loss += mae_loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            for batch_idx, ((fc, base), label) in enumerate(val_loader):
                fc = fc.to(device)  # move data to device
                base = base.to(device)
                label = label.to(device)

                output = model(fc, base)

                (mae_loss, l2_reg) = model.loss(output, label.view(output.shape))
                loss = mae_loss + l2_reg
                val_loss += mae_loss.item()

        # append losses to lists
        train_losses.append(train_loss / len(train_set))
        val_losses.append(val_loss / len(val_set))

        # save the model if the validation loss is at its minimum
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = (copy.deepcopy(model.state_dict()), epoch)

    model.eval()
    with torch.no_grad():
        for batch_idx, ((fc, base), label) in enumerate(test_loader):
            fc = fc.to(device)  # move data to device
            base = base.to(device)
            label = label.to(device)

            output = model(fc, base)

    with open(os.path.join(parent_dir, 'train_simple_net.txt'), 'a') as f:
        f.write(str(test_index) + ', ' + str(label.item()) + ', ' + str(output.item()) + '\n')


if __name__ == "__main__":
    main()
