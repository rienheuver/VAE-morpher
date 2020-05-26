import sys
import os
import dlib
import glob
import numpy as np
import cv2
import math


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


if len(sys.argv) < 3 or len(sys.argv) > 4:
    print(
        "3 or 4 arguments required:\n"
        "First give the python file (normalise.py)\n"
        "then the predictor.dat file\n"
        "then the faces directory\n"
        "and last argument is optional for debugging: --debug\n"
        "Example execution:\n"
        "python normalise.py shape_predictor_68.dat pictures/ --debug\n\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]
debug = len(sys.argv) >= 4 and sys.argv[3] == '--debug'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for index, f in enumerate(glob.glob(os.path.join(faces_folder_path, "*.jpg"))):
    destPath = ''
    if index % 100 == 2:
        destPath = './fullface/testing2/' + f.split('/')[1]
    else:
        destPath = './fullface/training2/' + f.split('/')[1]
    if os.path.exists(destPath):
        print("File already exists in output directory, skipping...")
        continue;
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        if shape.num_parts == 68:
            lefteyeX = int((shape.part(36).x + shape.part(39).x) / 2)
            lefteyeY = int((shape.part(36).y + shape.part(39).y) / 2)

            righteyeX = int((shape.part(42).x + shape.part(45).x) / 2)
            righteyeY = int((shape.part(42).y + shape.part(45).y) / 2)

            try:
                # Read image from the disk.
                img = cv2.imread(f)

                if debug:
                    cv2.imshow('result.jpg', img)
                    cv2.waitKey(0)

                # Shape of image in terms of pixels.
                (rows, cols) = img.shape[:2]

                # Rotate image (left and right eye on one line)
                dist = ((righteyeX - lefteyeX) ** 2 + (righteyeY -
                                                       lefteyeY) ** 2) ** 0.5  # distance between eyes
                if dist < 50:
                    print("Eyes are too close")
                    break
                heightDif = righteyeY - lefteyeY  # height difference between eyes
                # angle between right eye and horizontal
                angle = math.degrees(math.asin(heightDif / dist))

                M = cv2.getRotationMatrix2D((lefteyeX, lefteyeY), angle, 1)
                # rotate face to get left and right eyes on one line
                img = cv2.warpAffine(img, M, (cols, rows))

                if debug:
                    cv2.imshow('result.jpg', img)  # show turned face
                    cv2.waitKey(0)

                eyePoints = (righteyeX, righteyeY)
                newEyes = M.dot(np.array(eyePoints + (1,)))

                newRightEyeX = newEyes[0]
                newRightEyeY = newEyes[1]
                newLeftEyeX = lefteyeX
                newLeftEyeY = lefteyeY

                width = 160.0
                paddingY = 55
                paddingX = 55
                factor = (width - 2 * paddingX) / (newRightEyeX - newLeftEyeX)
                # resize face to get normalized eye distance
                img = cv2.resize(img, None, fx=factor, fy=factor)

                if debug:
                    cv2.imshow('result.jpg', img)  # show resized face
                    cv2.waitKey(0)

                # Move image (get eyes in the right positions)
                newLeftEyeX = newLeftEyeX * factor
                newLeftEyeY = newLeftEyeY * factor
                newRightEyeX = newRightEyeX * factor
                newRightEyeY = newRightEyeY * factor

                # cv2.circle(img, (int(newRightEyeX), int(newRightEyeY)), 20, (255,0,131))
                # cv2.circle(img, (int(newLeftEyeX), int(newLeftEyeY)), 20, (255,0,131))

                M = np.float32([[1, 0, -newLeftEyeX + paddingX],
                                [0, 1, -newLeftEyeY + paddingY]])
                img = cv2.warpAffine(img, M, (cols, rows))

                if debug:
                    cv2.imshow('result.jpg', img)  # show resized face
                    cv2.waitKey(0)

                img = img[0:int(width), 0:int(width)]
                if debug:
                    cv2.imshow('result.jpg', img)  # show resized face
                    cv2.waitKey(0)

                # cv2.imshow('result.jpg', img)
                cv2.imwrite(destPath, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            except IOError:
                print('Error while reading files !!!')
