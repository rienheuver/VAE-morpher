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
from matplotlib import lines as mlines
from matplotlib import transforms as mtransforms
import sys
import json

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

bestDistances = []

firstPicDir = 'normalised/testing/first-pic/'
firstPics = os.listdir(firstPicDir)
secondPicDir = 'normalised/testing/second-pic/'
secondPics = os.listdir(secondPicDir)
print(firstPics)

def getIdentity(file):
    return file.split('/')[-1].split('d')[0]

def findOther(file):
    identity = getIdentity(file)
    option = [s for s in secondPics if identity in s][0]
    option_file = secondPicDir + option
    option_img = face_recognition.load_image_file(option_file)
    option_enc = face_recognition.face_encodings(option_img)[0]
    return option, option_enc, option_img

def getOriginal(file, index, num):
    if os.path.exists(file):
        file_img = face_recognition.load_image_file(file)
        file_enc = face_recognition.face_encodings(file_img)[0]
        return file_img, file_enc
    else:
        print('This one is empty: '+str(index)+'.'+str(num))
        return np.array([]), np.empty(128)

fromImage = transforms.ToTensor()
def getSample(index):
    filename = firstPicDir + firstPics[index]
    face = fromImage(Image.open(filename))

    mu, logvar = model.encode(
        face.unsqueeze(0).to("cpu"))
    latent = model.reparameterize(mu, logvar, 0)
    reconstruct = model.decode(latent).cpu()
    rec_filename = 'temp/reconstruction.png'
    save_image(reconstruct, rec_filename)
    return filename, rec_filename, latent


def getEncodings(original, reconstruction):
    rec_image = face_recognition.load_image_file(reconstruction)
    rec_enc = face_recognition.face_encodings(rec_image)[0]

    original_image = face_recognition.load_image_file(original)
    original_enc = face_recognition.face_encodings(original_image)[0]

    return original_enc, rec_enc, original_image, rec_image


def distance(enc1, enc2):
    dist = face_recognition.face_distance([enc1], enc2)
    if len(dist) == 0:
        return 0
    return face_recognition.face_distance([enc1], enc2)[0]


R = "\033[0;31;40m"  # RED
G = "\033[0;32;40m"  # GREEN
N = "\033[0m"  # Reset


def threshColour(distance):
    if distance <= threshold:
        return G+str(distance)+N
    else:
        return R+str(distance)+N


source1_source2s = []
other1_other2s = []
rec1_rec2s = []
source1_rec1s = []
source2_rec2s = []
source1_other1s = []
source2_other2s = []
source1_morphs = []
source2_morphs = []
other1_morphs = []
other2_morphs = []
other1_rec1s = []
other2_rec2s = []
rec1_morphs = []
rec2_morphs = []
source1_other2s = []
source2_other1s = []
source1_rec2s = []
source2_rec1s = []
other1_rec2s = []
other2_rec1s = []

source1_orig1s = []
orig1_other1s = []
source2_orig2s = []
orig2_other2s = []

for i in range(0, 1):
    success = 0
    fails = 0
    double = 0
    counter = 0
    amountOfPeople = len(firstPics)
    for index in range(0, amountOfPeople):
        print('Identity ' + str(index+1) + '/' + str(amountOfPeople))
        for index2 in range(0, amountOfPeople): # index+2 or amountOfPeople, first is much shorter
            counter += 1
            if index == index2:
                continue
            try:
                source1, rec1, latent1 = getSample(index)
                # source1_enc, rec1_enc, source1_img, rec1_img = getEncodings(
                #     source1, rec1)
                other1, other1_enc, other1_img = findOther(source1)

                source2, rec2, latent2 = getSample(index2)
                # source2_enc, rec2_enc, source2_img, rec2_img = getEncodings(
                #     source2, rec2)
                other2, other2_enc, other2_img = findOther(source2)

                latent_morph = (latent1 + latent2) / 2
                rec_morph = model.decode(latent_morph).cpu()
                morph_filename = 'temp/morph-seeker.png'
                save_image(rec_morph, morph_filename)
                morph_image = face_recognition.load_image_file(morph_filename)
                morph_enc = face_recognition.face_encodings(morph_image)[0]

                ### HOLY GRAIL ###
                grail1 = distance(other1_enc, morph_enc)
                grail2 = distance(other2_enc, morph_enc)
                other1_morphs.append(grail1)
                other2_morphs.append(grail2)

                otherother = distance(other1_enc, other2_enc)
                other1_other2s.append(otherother)
                ### END OF HOLY GRAIL ###
                # source1_source2s.append(distance(source1_enc, source2_enc))
                # recrec = rec1_rec2s.append(distance(rec1_enc, rec2_enc))
                # source1_rec1s.append(distance(source1_enc, rec1_enc))
                # source2_rec2s.append(distance(source2_enc, rec2_enc))
                # source1_other1s.append(distance(source1_enc, other1_enc))
                # source2_other2s.append(distance(source2_enc, other2_enc))
                # source1_morphs.append(distance(source1_enc, morph_enc))
                # source2_morphs.append(distance(source2_enc, morph_enc))
                # other1_rec1s.append(distance(other1_enc, rec1_enc))
                # other2_rec2s.append(distance(other2_enc, rec2_enc))
                # rec1_morphs.append(distance(rec1_enc, morph_enc))
                # rec2_morphs.append(distance(rec2_enc, morph_enc))
                # source1_other2s.append(distance(source1_enc, other2_enc))
                # source2_other1s.append(distance(source2_enc, other1_enc))
                # source1_rec2s.append(distance(source1_enc, rec2_enc))
                # source2_rec1s.append(distance(source2_enc, rec1_enc))
                # other1_rec2s.append(distance(other1_enc, rec2_enc))
                # other2_rec1s.append(distance(other2_enc, rec1_enc))

                if grail1 <= threshold and grail2 <= threshold and otherother <= threshold:
                    print('Lookalike success: ' + getIdentity(source1) + ', ' + getIdentity(source2))
                    print(grail1, grail2, otherother)
                    success += 1
                    # pil = Image.fromarray(np.hstack((source1_img, rec1_img, morph_image, rec2_img, source2_img)))
                    # pil.show()

            #   if other1_morph <= threshold:
            #       success += 1
            #   else:
            #       fails += 1

            #   if other2_morph <= threshold:
            #       success += 1
            #   else:
            #       fails += 1

            #   if other1_morph <= threshold and other2_morph <= threshold:
            #       double += 1
            except:
                print('kapot')
            # print(str(counter) + '/' + str((amountOfPeople*(amountOfPeople-1))/2))
    print('Successful morphs: ' + str(success))
    # dataTable = [
    #           ["",                "Source1",                      "Other1",                     "Reconstruction1",                "Morph",                      "Source2",                      "Other2",                     "Reconstruction2"],
    #           ["Source1",         threshColour(0),                threshColour(sum(source1_other1s)/len(source1_other1s)), threshColour(sum(source1_rec1s)/len(source1_rec1s)),       threshColour(sum(source1_morphs)/len(source1_morphs)),  threshColour(sum(source1_source2s)/len(source1_source2s)),  threshColour(sum(source1_other2s)/len(source1_other2s)), threshColour(sum(source1_rec2s)/len(source1_rec2s))],
    #           ["Other1",          '-',                            threshColour(0),              threshColour(sum(other1_rec1s)/len(other1_rec1s)),        threshColour(sum(other1_morphs)/len(other1_morphs)),   threshColour(sum(source2_other1s)/len(source2_other1s)),   threshColour(sum(other1_other2s)/len(other1_other2s)),  threshColour(sum(other1_rec2s)/len(other1_rec2s))],
    #           ["Reconstruction1", '-','-',                                                      threshColour(0),                  threshColour(sum(rec1_morphs)/len(rec1_morphs)),     threshColour(sum(source2_rec1s)/len(source2_rec1s)),     threshColour(sum(other2_rec1s)/len(other2_rec1s)),    threshColour(sum(rec1_rec2s)/len(rec1_rec2s))],
    #           ["Morph",           '-','-','-',                                                                                    threshColour(0),              threshColour(sum(source2_morphs)/len(source2_morphs)),    threshColour(sum(other2_morphs)/len(other2_morphs)),   threshColour(sum(rec2_morphs)/len(rec2_morphs))],
    #           ["Source2",         '-','-','-','-',                                                                                                              threshColour(0),                threshColour(sum(source2_other2s)/len(source2_other2s)), threshColour(sum(source2_rec2s)/len(source2_rec2s))],
    #           ["Other2",          '-','-','-','-','-',                                                                                                                                          threshColour(0),              threshColour(sum(other2_rec2s)/len(other2_rec2s))],
    #           ["Reconstruction2", '-','-','-','-','-','-',                                                                                                                                                                    threshColour(0)]
    #       ]
    # print(tabulate(dataTable, headers="firstrow", tablefmt="fancy_grid"))

    histBins = np.linspace(0,1,41)
    plt.plot(other1_other2s, other1_morphs, '.', label='Subject A - Morph')
    plt.plot(other1_other2s, other2_morphs, '.', label='Subject B - Morph')
    plt.xlabel('Subject A - Subject B')
    plt.legend(loc='upper left')
    plt.show()

    # plt.hist(bestDistances, bins=histBins, rwidth=0.8)
    # plt.title('bestDistances')
    # plt.show()

    # print('source1_other1s: (should be 0.3 or so)')
    # print(threshColour(sum(source1_other1s)/len(source1_other1s)))

    # print('source2_other2s: (should be 0.3 or so)')
    # print(threshColour(sum(source2_other2s)/len(source2_other2s)))


    # Genuine = source - other
    # morph = morph - other
    # imposter = source2 - other
    # plt.hist(source1_other1s, bins=histBins, rwidth=0.8, label='Genuine', alpha=0.5)
    # plt.hist(other1_morphs, bins=histBins, rwidth=0.8, label='Morphs', alpha=0.5)
    # plt.hist(source2_other1s, bins=histBins, rwidth=0.8, label='Imposter', alpha=0.5)
    # plt.legend(loc='upper right')
    # plt.title('Morph analysis')
    # plt.show()

    # plt.hist(source1_source2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source1_source2s')
    # plt.show()
    # plt.hist(other1_other2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other1_other2s')
    # plt.show()
    # plt.hist(rec1_rec2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('rec1_rec2s')
    # plt.show()
    # plt.hist(source1_rec1s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source1_rec1s')
    # plt.show()
    # plt.hist(source2_rec2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source2_rec2s')
    # plt.show()
    # plt.hist(source1_other1s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source1_other1s')
    # plt.show()
    # plt.hist(source2_other2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source2_other2s')
    # plt.show()
    # plt.hist(source1_morphs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source1_morphs')
    # plt.show()
    # plt.hist(source2_morphs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source2_morphs')
    # plt.show()
    
    #### HOLY GRAIL ####
    # plt.hist(other1_morphs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other1_morphs')
    # plt.show()
    # plt.hist(other2_morphs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other2_morphs')
    # plt.show()
    ### END OF HOLY GRAIL
    # plt.hist(other1_rec1s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other1_rec1s')
    # plt.show()
    # plt.hist(other2_rec2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other2_rec2s')
    # plt.show()
    # plt.hist(rec1_morphs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('rec1_morphs')
    # plt.show()
    # plt.hist(rec2_morphs, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('rec2_morphs')
    # plt.show()
    # plt.hist(source1_other2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source1_other2s')
    # plt.show()
    # plt.hist(source2_other1s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source2_other1s')
    # plt.show()
    # plt.hist(source1_rec2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source1_rec2s')
    # plt.show()
    # plt.hist(source2_rec1s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('source2_rec1s')
    # plt.show()
    # plt.hist(other1_rec2s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other1_rec2s')
    # plt.show()
    # plt.hist(other2_rec1s, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rwidth=0.8)
    # plt.title('other2_rec1s')
    # plt.show()