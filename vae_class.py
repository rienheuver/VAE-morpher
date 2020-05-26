from torch import nn, optim
import torch
from skimage.util import random_noise
import matplotlib.pyplot as plt

def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']+1, checkpoint['bce_losses'], checkpoint['kld_losses'], checkpoint['detail_losses']

class VAE(nn.Module):
    def __init__(self, max_channels, latent_size, img_size):
        super(VAE, self).__init__()
        self.max_channels = max_channels
        self.latent_size = latent_size
        self.img_size = img_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Output size is 1x1 image with 128 max_channels => 128 nodes

        self.fully_encode_1 = nn.Linear(256, 256)
        self.fully_encode_2 = nn.Linear(256, 256)

        # self.fully_decode = nn.Linear(128, sli_size**2*max_channels)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,
                               256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256,
                               128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # plt.imshow(x[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        noise = torch.tensor(random_noise(x.cpu(), mode="gaussian", mean=0, var=0.005, clip=True))
        # plt.imshow(noise[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()
        h1 = self.encoder(x)
        flat = h1.view(h1.size(0), -1)
        mu, logvar = self.fully_encode_1(flat), self.fully_encode_2(flat)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps = None):
        std = torch.exp(0.5*logvar)
        if (eps == None):
            eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, latent):
        # h2 = self.fully_decode(latent)
        h2 = latent
        unflat = h2.view(h2.size(0), self.latent_size, 1, 1)
        return self.decoder(unflat)

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent).view(-1, self.img_size**2), mu, logvar