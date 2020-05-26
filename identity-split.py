import sys
import os
import glob
from PIL import Image

if len(sys.argv) < 3:
    print('Too few arguments')
    sys.exit()

firstDir = sys.argv[1]
secondDir = sys.argv[2]
counter = 0
savedIdentities = []
for index, f in enumerate(glob.glob(os.path.join(firstDir, "*.jpg"))):
    identity = f.split('/')[1].split('d')[0]
    if identity in savedIdentities:
        print('already got this one')
        continue
    match = False
    for index, f2 in enumerate(glob.glob(os.path.join(secondDir, "*.jpg"))):
        identity2 = f2.split('/')[1].split('d')[0]
        if identity == identity2:
            match = f2
    if match:
        counter += 1
        newF = os.path.join('testing', 'first-pic', f.split('/')[1])
        newMatch = os.path.join('testing', 'second-pic', match.split('/')[1])

        if counter <= 200:
            os.rename(f, newF)
            os.rename(match, newMatch)

        savedIdentities.append(identity)
    else:
        newLocation = os.path.join('training', f.split('/')[1])
        os.rename(f, newLocation)
    if counter == 200:
        print('Found 200 matches')