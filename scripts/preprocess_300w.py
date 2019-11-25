import os
import argparse
from skimage import io
from PIL import Image
import numpy as np
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the 300w folder')
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

output_path = args.output
load_path = args.path
##
landmark_image_save_path = os.path.join(output_path,'test')
if not os.path.exists(landmark_image_save_path):
    os.mkdir(landmark_image_save_path)
p = 0.3
p2 = 0.2

train_folder_list = ['afw','helen/trainset','lfpw/trainset']
test_folder_list =['helen/testset','lfpw/testset','ibug']

train_images_path = os.path.join(output_path, 'train_images')
test_images_path = os.path.join(output_path, 'test_images')

train_landmarks_path = os.path.join(output_path, 'train_landmarks')
test_landmarks_path = os.path.join(output_path,'test_landmarks')

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(train_images_path):
    os.mkdir(train_images_path)
if not os.path.exists(test_images_path):
    os.mkdir(test_images_path)
if not os.path.exists(train_landmarks_path):
    os.mkdir(train_landmarks_path)
if not os.path.exists(test_landmarks_path):
    os.mkdir(test_landmarks_path)


def handle(folder_list, images_save_path, landmarks_save_path):
    for subfolder in folder_list:
        filenames = os.listdir(os.path.join(load_path, subfolder))
        for filename in filenames:
            if not filename[-3:] == 'jpg' and not filename[-3:] == 'png':
                continue

            image = io.imread(os.path.join(load_path, subfolder, filename))
            if len(image.shape) == 2:
                L_image = Image.open(os.path.join(load_path, subfolder, filename))
                out = L_image.convert("RGB")
                image = np.array(out)

            with open(os.path.join(load_path, subfolder, (filename[:-3] + 'pts')), 'r') as f:
                landmark_load_str = f.readlines()[3:-1]

            landmark_load = np.ones((68, 2))
            for i in range(len(landmark_load_str)):
                t = landmark_load_str[i].split()
                landmark_load[i, 0] = float(t[0])
                landmark_load[i, 1] = float(t[1])

            X1 = np.min(landmark_load[:, 0])
            X2 = np.max(landmark_load[:, 0])
            Y1 = np.min(landmark_load[:, 1])
            Y2 = np.max(landmark_load[:, 1])

            H_b = Y2 - Y1
            W_b = X2 - X1

            X1_c = X1 - p * W_b
            X2_c = X2 + p * W_b
            Y1_c = Y1 - p * H_b - p2 * H_b
            Y2_c = Y2 + p * H_b - p2 * H_b

            W_c = X2_c - X1_c
            H_c = Y2_c - Y1_c

            if W_c < H_c:
                gap = H_c - W_c
                X1_c -= gap / 2
                X2_c += gap / 2

            else:
                gap = W_c - H_c
                Y1_c -= gap / 2
                Y2_c += gap / 2

            Y1_c = int(Y1_c + 0.5)
            Y2_c = int(Y2_c + 0.5)
            X1_c = int(X1_c + 0.5)
            X2_c = int(X2_c + 0.5)

            W_c = X2_c - X1_c
            H_c = Y2_c - Y1_c

            H_c = int(H_c + 0.5)
            W_c = int(W_c + 0.5)

            image_c = np.full((H_c, W_c, 3), 255)

            H, W = image.shape[0:2]

            X_startpos = Y_startpos = 0
            X_endpos, Y_endpos = W_c, H_c

            if X1_c < 0:
                X_startpos = -X1_c
                X1_c = 0
            if Y1_c < 0:
                Y_startpos = -Y1_c
                Y1_c = 0

            if Y2_c >= H:
                Y_endpos = H_c - (Y2_c - H)
                Y2_c = H
            if X2_c >= W:
                X_endpos = W_c - (X2_c - W)
                X2_c = W

            image_c[:, 0:X_startpos] = 0
            image_c[0:Y_startpos, :] = 0
            image_c[Y_startpos:Y_endpos, X_startpos:X_endpos] = image[Y1_c:Y2_c, X1_c:X2_c]
            image_c[:, X_endpos:W_c] = 0
            image_c[Y_endpos:H_c, :] = 0

            image_save = scipy.misc.imresize(image_c, (256, 256))

            scale = 256.0 / H_c
            landmark_save = np.ones((68, 2))
            landmark_save[:, 0] = ((landmark_load[:, 0] + X_startpos - X1_c) * scale + 0.5)
            landmark_save[:, 1] = ((landmark_load[:, 1] + Y_startpos - Y1_c) * scale + 0.5)
            landmark_save = landmark_save.astype(int)
            io.imsave(os.path.join(images_save_path, filename[:-4] + '.png'), image_save)

            with open(os.path.join(landmarks_save_path, (filename[:-3] + 'txt')), 'w') as f2:
                for i in range(landmark_save.shape[0]):
                    f2.write(str(landmark_save[i, 0]) + ' ' + str(landmark_save[i, 1]) + '\n')

            landmark_image = image_save
            landmark_image[landmark_save[:, 1], landmark_save[:, 0]] = 255
            ##
            io.imsave(os.path.join(landmark_image_save_path,filename),landmark_image)

            print(filename)


handle(train_folder_list,train_images_path,train_landmarks_path)
handle(test_folder_list, test_images_path, test_landmarks_path)

