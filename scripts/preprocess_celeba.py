import os
import argparse
from skimage import io
from scipy.misc import imresize

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the celeba img_align_celeba folder')
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

output_path = args.output
dataset_path = args.path

if not os.path.exists(output_path):
    os.mkdir(output_path)

train_path = os.path.join(output_path,'celeba_train_images')
test_path = os.path.join(output_path,'celeba_test_images')
val_path = os.path.join(output_path,'celeba_val_images')

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)
if not os.path.exists(val_path):
    os.mkdir(val_path)


filenames = os.listdir(dataset_path)

for filename in filenames:
    index = int(filename[:-4])
    if index <= 162770:
        save_path = train_path
    elif index <= 182637:
        save_path = val_path
    else:
        save_path = test_path

    img = io.imread(os.path.join(dataset_path,filename))

    h,w,c = img.shape

    border = (h-w)//2
    img = img[border:-border,...]

    img = imresize(img,(256,256))

    io.imsave(os.path.join(save_path,filename[:-4]+'.png'),img)

    print(filename)





