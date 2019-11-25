import os
import argparse
from skimage import io
import numpy as np
import scipy


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the WFLW_images folder')
parser.add_argument('--output', type=str, help='path to the output folder')
parser.add_argument('--annotation_path', type=str, help='path to the WFLW_annotations folder')
args = parser.parse_args()

output_path = args.output
image_load_path = args.path
annotation_path = args.annotation_path

annotation_load_paths = [os.path.join(annotation_path,'list_98pt_rect_attr_train_test','list_98pt_rect_attr_train.txt'), os.path.join(annotation_path,'list_98pt_rect_attr_train_test','list_98pt_rect_attr_test.txt')]
image_save_paths=[os.path.join(output_path,'train_images'),os.path.join(output_path,'test_images')]
landmark_save_paths = [os.path.join(output_path,'train_landmarks'), os.path.join(output_path, 'test_landmarks')]

landmark_image_save_path = os.path.join(output_path,'visualize')

p = 0.3
p2 = 0.2

def handle(annotation_load_path, image_save_path, landmark_save_path):
    count = 1
    if not os.path.exists(landmark_save_path):
        os.mkdir(landmark_save_path)

    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)

    if not os.path.exists(landmark_image_save_path):
        os.mkdir(landmark_image_save_path)

    with open(annotation_load_path, 'r') as anno:
        lines = anno.readlines()
        print(len(lines))

    for line in lines:
        divides = line.split(' ')
        img_path = os.path.join(image_load_path, divides[-1][:-1])
        image = io.imread(img_path)
        filename = img_path.split('/')[-1]

        landmark_load = np.array(divides[:196]).reshape((98, 2)).astype('float')

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
            # print('yes')
            X_endpos = W_c - (X2_c - W)
            X2_c = W

        image_c[:, 0:X_startpos] = 0
        image_c[0:Y_startpos, :] = 0
        image_c[Y_startpos:Y_endpos, X_startpos:X_endpos] = image[Y1_c:Y2_c, X1_c:X2_c]
        image_c[:, X_endpos:W_c] = 0
        image_c[Y_endpos:H_c, :] = 0

        image_save = scipy.misc.imresize(image_c, (256, 256))

        scale = 256.0 / H_c
        landmark_save = np.ones((98, 2))
        landmark_save[:, 0] = ((landmark_load[:, 0] + X_startpos - X1_c) * scale + 0.5)
        landmark_save[:, 1] = ((landmark_load[:, 1] + Y_startpos - Y1_c) * scale + 0.5)
        landmark_save = landmark_save.astype(int)
        if os.path.exists(os.path.join(image_save_path, filename[:-4]+'.png')):
            print('repeat!')
            filename = filename[:-4] + '_' + str(count) + filename[-4:]
        io.imsave(os.path.join(image_save_path, filename[:-4]+'.png'), image_save)

        with open(os.path.join(landmark_save_path, (filename[:-3] + 'txt')), 'w') as f2:
            for i in range(landmark_save.shape[0]):
                f2.write(str(landmark_save[i, 0]) + ' ' + str(landmark_save[i, 1]) + '\n')

        landmark_image = image_save
        landmark_image[landmark_save[:, 1], landmark_save[:, 0]] = 255
        io.imsave(os.path.join(landmark_image_save_path, filename), landmark_image)
        print(filename + str(count))
        count = count + 1

handle(annotation_load_paths[0],image_save_paths[0],landmark_save_paths[0])
handle(annotation_load_paths[1],image_save_paths[1],landmark_save_paths[1])