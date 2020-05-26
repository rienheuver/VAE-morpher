import face_recognition
import vae_class
import torchvision.transforms as transforms
from torch import optim
from torchvision.utils import save_image
from PIL import Image
import sys
import numpy as np

img_size = 160
max_channels = 256
latent_size = 256

if len(sys.argv) < 3:
    print("2 arguments needed, namely the two images to morph, optionally as third argument specify the output-file")
    sys.exit(0)

modelFile = 'model.pt'
model = vae_class.VAE(max_channels, latent_size, img_size).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model, optimizer, start_epoch, _, _, _ = vae_class.load_ckp(
    modelFile, model, optimizer)

fromImage = transforms.ToTensor()
def makeMorph(file1, file2, out):
    face1 = fromImage(Image.open(file1))
    face2 = fromImage(Image.open(file2))

    mu1, logvar1 = model.encode(
        face1.unsqueeze(0).to("cpu"))
    latent1 = model.reparameterize(mu1, logvar1, 0)
    mu2, logvar2 = model.encode(
        face2.unsqueeze(0).to("cpu"))
    latent2 = model.reparameterize(mu2, logvar2, 0)

    latent_morph = (latent1 + latent2) / 2
    rec_morph = model.decode(latent_morph).cpu()
    save_image(rec_morph, out)


file1 = sys.argv[1]
file2 = sys.argv[2]
out = 'result.jpg'
if len(sys.argv) > 3:
    out = sys.argv[3]
makeMorph(file1, file2, out)

# Disable below lines to not display the morph but only save it
file1_img = face_recognition.load_image_file(file1)
file2_img = face_recognition.load_image_file(file2)
vae_img = face_recognition.load_image_file(out)

pil = Image.fromarray(np.hstack((file1_img, vae_img, file2_img)))
pil.show()