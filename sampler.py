import vae_class
import torch
from torch import nn, optim
from torchvision.utils import save_image
import math

img_size = 160
max_channels = 256
latent_size = 256
sample_count = 16

if math.sqrt(sample_count) % 1 != 0:
    print('Sample count needs to be a square')
    exit()

model = vae_class.VAE(max_channels, latent_size, img_size).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model, optimizer, start_epoch, _, _ = vae_class.load_ckp(
    'noisy.pt', model, optimizer)

if __name__ == "__main__":
    row = None
    matrix = None
    for i in range(1, sample_count+1):
        randomLatent = torch.rand((1, latent_size))
        result = model.decode(randomLatent).cpu()
        result.view(3, img_size, img_size)
        if row is None:
            row = result    
        else:
            row = torch.cat((row, result), 2)
        if i % math.sqrt(sample_count) == 0:
            if matrix is None:
                matrix = row
            else:
                matrix = torch.cat((matrix, row), 3)
            row = None
    save_image(matrix, 'fullface/vae-samples/'+str(sample_count)+'-samples.png')
