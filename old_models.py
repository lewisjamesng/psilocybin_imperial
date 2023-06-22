import torch.nn as nn
import torch

class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.0):
        super(SimpleFCNN, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 2):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # Output layer
        self.output_layer_1 = nn.Linear(hidden_sizes[-2] + 2, hidden_sizes[-1])
        self.output_layer_2 = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x, base):        
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.output_layer_1(torch.cat([x, base], dim=1))
        x = self.relu(x)
        x = self.output_layer_2(x)
        return x
    
    def loss(self, true, pred):
        reg_loss = 0.
        for param in self.parameters():
            reg_loss += torch.linalg.norm(param)
        
        mae_loss = nn.L1Loss(reduction='sum')(true, pred)
        return (mae_loss, reg_loss)



class Decoder(nn.Module):
    def __init__(self, nf, nc):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.Tanh(),

            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.Tanh(),

            nn.ConvTranspose2d(nf * 4, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.Tanh(),
            
            nn.ConvTranspose2d(nf, nf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(nf),
            nn.Tanh(),

            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=True),
        )
    def forward(self, input):
        return self.layers(input)

class Encoder(nn.Module):
    def __init__(self, nf, nc):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.Tanh(),
           
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.Tanh(),
           
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.Tanh(),
            
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.Tanh(),
            
            nn.Conv2d(nf * 4, nf * 4, 6, 1, 0, bias=False),
            nn.Tanh(),
        )
    def forward(self, input):
        return self.layers(input)
    
class AutoEncoder(nn.Module):
    def __init__(self, nf, nc):
        super(AutoEncoder, self).__init__()
        self.decoder = Decoder(nf, nc)
        self.encoder = Encoder(nf, nc)

    def forward(self, input):
        out = self.encoder(input)        
        return self.decoder(out)

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),

            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=True),
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.Tanh(),
           
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.Tanh(),
           
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Tanh(),
            
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Tanh(),
            
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.Tanh(),
            
            nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)