import face_recognition
import torch
import torchvision
import torchvision.transforms as transforms
import vae_class
from torch import optim
from torchvision.utils import save_image
from PIL import Image, ImageFilter
import numpy as np
import os
from tabulate import tabulate
from matplotlib import pyplot as plt
import sys

img_size = 160
max_channels = 256
latent_size = 256
threshold = 0.6

modelFile = 'model.pt'
if len(sys.argv) > 1:
    modelFile = sys.argv[1]
model = vae_class.VAE(max_channels, latent_size, img_size).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model, optimizer, start_epoch, _, _, _ = vae_class.load_ckp(
    modelFile, model, optimizer)

def getEncodings(original):
    original_image = face_recognition.load_image_file(original)
    original_enc = face_recognition.face_encodings(original_image)[0]

    return original_enc, original_image

def distance(enc1, enc2):
    dist = face_recognition.face_distance([enc1], enc2)
    if len(dist) == 0:
        return 0
    return face_recognition.face_distance([enc1], enc2)[0]

fromImage = transforms.ToTensor()
def makeMorph(file1, file2):
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
    morph_filename = 'temp/vae-morph.png'
    save_image(rec_morph, morph_filename)

firstPicDir = 'normalised/testing/first-pic/'
firstPics = os.listdir(firstPicDir)
amountOfPeople = len(firstPics)

distances = []
for index in range(0, amountOfPeople):
    for index2 in range(index+1, min(amountOfPeople, index+4)): # small test: change max into min
        try:
            file1 = firstPicDir + firstPics[index]
            file2 = firstPicDir + firstPics[index2]
    # file1 = firstPicDir + '04493d34.jpg'
    # file2 = firstPicDir + '04344d39.jpg'
            cmd = "python ../face_morpher/facemorpher/morpher.py --src=" + file1 + " --dest=" + file2 + " --background=average --num=5 --out_frames=temp --width=160 --height=160"
            os.system(cmd)
            makeMorph(file1, file2)

            landmark = 'temp/frame002.png'
            vae = 'temp/vae-morph.png'

            landmark_enc, landmark_img = getEncodings(landmark)
            vae_enc, vae_img = getEncodings(vae)

            # _, file1_img = getEncodings(file1)
            # _, file2_img = getEncodings(file2)
            # pil = Image.fromarray(np.hstack((file1_img, file2_img, landmark_img, vae_img)))
            # pil.show()

            distances.append(distance(landmark_enc, vae_enc))
            print('Identity ' + str(index+1) + ' out of ' + str(amountOfPeople))
            print(str(len(distances)) + '/' + str((amountOfPeople*(amountOfPeople-1))/2))
        except:
            print('kapot')

print(sum(distances) / len(distances))
histBins = np.linspace(0,1,41)
plt.hist(distances, bins=histBins, rwidth=0.8)
plt.title('Landmark to VAE morph distances')
plt.show()