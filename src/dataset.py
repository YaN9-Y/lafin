import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.misc import imresize
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, landmark_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)
        self.landmark_data = self.load_flist(landmark_flist)

        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])

        if self.config.MODEL != 3:
            landmark = self.load_lmk([size, size], index, img.shape)
        else: ## test on stage 3 doesn't need ground truth landmarks
            landmark = np.zeros((68,2))

        if self.config.AUGMENTATION_TRAIN == 1 and self.config.MODEL == 1:
            landmark_orig = landmark.copy()
        if self.config.AUGMENTATION_TRAIN == 1 and self.config.MODEL == 1:
            img_orig = img.copy()

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size, centerCrop=True)

        # load mask
        mask = self.load_mask(img, index)

        if self.config.AUGMENTATION_TRAIN == 1 and self.config.MODEL == 1:
            temp = self.mask
            self.mask = 1
            mask2 = self.load_mask(img_orig, index)
            self.mask = temp

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            landmark[:, 0] = self.input_size - landmark[:, 0]
            landmark = self.shuffle_lr(landmark)

            mask = mask[:, ::-1, ...]
            if self.config.AUGMENTATION_TRAIN == 1 and self.config.MODEL == 1:
                mask2 = mask2[:,::-1,...]

        if self.augment and self.config.MODEL == 1 and np.random.uniform(0,1)<=0.2:
            img = imresize(img,[int(self.config.INPUT_SIZE*3/8),int(self.config.INPUT_SIZE*3/8)])
            img = imresize(img,[self.config.INPUT_SIZE, self.config.INPUT_SIZE])

        if self.augment and self.config.MODEL == 1:
            for i in range(3):
                img[i] = (img[i]*np.random.uniform(0.7,1.3))
                img[i][img[i]>1] = 1

        if self.config.MODEL == 1 and self.config.AUGMENTATION_TRAIN == 1 and self.config.MODE == 1:
            return self.to_tensor(img), torch.from_numpy(landmark).long(), self.to_tensor(mask), self.to_tensor(mask2), self.to_tensor(img_orig), torch.from_numpy(landmark_orig).long()
        else:
            return self.to_tensor(img), torch.from_numpy(landmark).long(), self.to_tensor(mask)



    def load_lmk(self, target_shape, index, size_before, center_crop = True):

        imgh,imgw = target_shape[0:2]
        landmarks = np.genfromtxt(self.landmark_data[index])
        landmarks = landmarks.reshape(self.config.LANDMARK_POINTS, 2)

        if self.input_size != 0:
            if center_crop:
                side = np.minimum(size_before[0],size_before[1])
                i = (size_before[0] - side) // 2
                j = (size_before[1] - side) // 2
                landmarks[0:self.config.LANDMARK_POINTS , 0] -= j
                landmarks[0:self.config.LANDMARK_POINTS , 1] -= i

            landmarks[0:self.config.LANDMARK_POINTS ,0] *= (imgw/side)
            landmarks[0:self.config.LANDMARK_POINTS ,1] *= (imgh/side)
        landmarks = (landmarks+0.5).astype(np.int16)

        return landmarks


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # 50% no mask, 25% random block mask, 25% external mask, for landmark predictor training.
        if mask_type == 5:
            mask_type = 0 if np.random.uniform(0,1) >= 0.5 else 4

        # no mask
        if mask_type == 0:
            return np.zeros((self.config.INPUT_SIZE,self.config.INPUT_SIZE))

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # center mask
        if mask_type == 2:
            return create_mask(imgw, imgh, imgw//2, imgh//2, x = imgw//4, y = imgh//4)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index%len(self.mask_data)])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]
        
        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def shuffle_lr(self, parts, pairs=None):
        """Shuffle the points left-right according to the axis of symmetry
        of the object.
        Arguments:
            parts {torch.tensor} -- a 3D or 4D object containing the
            heatmaps.
        Keyword Arguments:
            pairs {list of integers} -- [order of the flipped points] (default: {None})
        """

        if pairs is None:
            if self.config.LANDMARK_POINTS == 68:
                pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                     26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                     34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                     40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                     62, 61, 60, 67, 66, 65]
            elif self.config.LANDMARK_POINTS == 98:
                pairs = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
                         8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39,
                         38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67,
                         66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]

        if len(parts.shape) == 3:
            parts = parts[:,pairs,...]
        else:
            parts = parts[pairs,...]

        return parts

