import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import LandmarkDetectorModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class Lafin():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'landmark'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'landmark_inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.landmark_model = LandmarkDetectorModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        #train mode
        if self.config.MODE == 1:
            if self.config.MODEL == 1 :
                self.train_dataset = Dataset(config,config.TRAIN_LANDMARK_IMAGE_FLIST, config.TRAIN_LANDMARK_LANDMARK_FLIST, config.TRAIN_MASK_FLIST, augment=True,training=True)
                self.val_dataset = Dataset(config,config.TEST_LANDMARK_IMAGE_FLIST,config.TEST_LANDMARK_LANDMARK_FLIST, config.TEST_MASK_FLIST, augment=True, training=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

            elif self.config.MODEL == 2:

                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_INPAINT_LANDMARK_FLIST,
                                             config.TRAIN_MASK_FLIST, augment=True, training=True)
                self.val_dataset = Dataset(config, config.VAL_INPAINT_IMAGE_FLIST, config.VAL_INPAINT_LANDMARK_FLIST,
                                           config.TEST_MASK_FLIST, augment=True, training=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)


        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 1:
                self.test_dataset = Dataset(config, config.TEST_LANDMARK_IMAGE_FLIST, config.TEST_LANDMARK_LANDMARK_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)

            else:
                self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_INPAINT_LANDMARK_FLIST, config.TEST_MASK_FLIST,
                                            augment=False, training=False)


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            if self.config.AUGMENTATION_TRAIN == 1:
                self.inpaint_model.load()
            self.landmark_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.landmark_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.landmark_model.save()

        elif self.config.MODEL == 2:
            self.inpaint_model.save()


    def cal_weighted_mae(self, images1, images2, masks):
        '''
        :param images1:[B,C,H,W]
        :param images2: [B,C,H,W]
        :return:
        '''

        mae = 0
        for i in range(images1.shape[0]):
            mae += self.cal_mae(images1[i],images2[i]) / torch.sum(masks[i])

        mae = mae / images1.shape[0]

        return mae

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])


            for items in train_loader:

                self.inpaint_model.train()
                self.landmark_model.train()

                if model == 1 and self.config.AUGMENTATION_TRAIN == 1:
                    images, landmarks, masks, masks2, images_orig, landmarks_orig = self.cuda(*items)
                else:
                    images, landmarks, masks = self.cuda(*items)

                landmarks[landmarks >= self.config.INPUT_SIZE] = self.config.INPUT_SIZE - 1
                landmarks[landmarks < 0] = 0
                # edge model
                if model == 1:
                    # train

                    landmarks_output, loss, logs = self.landmark_model.process(images,masks,landmarks)

                    # backward
                    self.landmark_model.backward(loss)
                    iteration = self.landmark_model.iteration


                    if self.config.AUGMENTATION_TRAIN == 1:
                        self.inpaint_model.eval()
                        landmark_map = self.generate_landmark_map(landmarks_orig)
                        images_gen = self.inpaint_model(images_orig, landmark_map, masks2).detach()

                        if self.train_dataset.augment == True:
                            if self.config.CROP_INPUT == 1:
                                X1,X2,Y1,Y2 = torch.min(landmarks_orig[...,0],dim=1),torch.max(landmarks_orig[...,0],dim=1),torch.min(landmarks_orig[...,1],dim=1),torch.max(landmarks_orig[...,1],dim=1)
                                for i in range(landmarks_orig.shape[0]):
                                    landmarks_orig[i,:,0]=landmarks_orig[i,:,0]-Y1[i]
                                    landmarks_orig[i,:,1]=landmarks_orig[i,:,0]-X1[i]
                                scale = float(images.shape[2])/images_orig.shape[2]
                                landmarks_orig = (landmarks_orig * scale + 0.5).long()
                                images_gen = images_gen[:,:,Y1:Y2,X1:X2]
                                images_gen = F.interpolate(images_gen,[images.shape[2],images.shape[3]],mode='bilinear')

                            if np.random.binomial(1,0.5) > 0:
                                images_gen = torch.flip(images_gen,dims=[3])
                                landmarks_orig[:,:,0 ] = self.config.INPUT_SIZE - landmarks_orig[:,:,0]
                                landmarks_orig = self.train_dataset.shuffle_lr(landmarks_orig)
                            if np.random.uniform(0,1) <= 0.2:
                                images_gen = F.interpolate(images_gen,[int(self.config.INPUT_SIZE*3/8),int(self.config.INPUT_SIZE*3/8)],mode='bilinear')
                                images_gen = F.interpolate(images_gen,[self.config.INPUT_SIZE,self.config.INPUT_SIZE], mode='bilinear')
                            for i in range(3):
                                images_gen[:,i,...] = images_gen[:,i,...] * np.random.uniform(0.7,1.3)

                            images_gen[images_gen>1] = 1
			
                        landmarks_output2 , loss, logs2 = self.landmark_model.process_aug(images_gen, masks, landmarks_orig)
                        self.landmark_model.backward(loss)
                        self.landmark_model.iteration += 1
                        iteration = self.landmark_model.iteration
                        logs = logs + logs2


                # inpaint model

                elif model == 2:
                    landmarks[landmarks>=self.config.INPUT_SIZE] = self.config.INPUT_SIZE-1
                    landmarks[landmarks<0] = 0

                    landmark_map = torch.zeros((self.config.BATCH_SIZE,1,self.config.INPUT_SIZE,self.config.INPUT_SIZE)).cuda()

                    for i in range(landmarks.shape[0]):
                        landmark_map[i,0,landmarks[i,0:self.config.LANDMARK_POINTS,1],landmarks[i,0:self.config.LANDMARK_POINTS,0]] = 1

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images,landmark_map,masks)
                    outputs_merged = (outputs * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)
                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()
                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0 and self.config.MODEL == 2:
                    print('\nstart eval...\n')
                    self.eval()
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.landmark_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, landmarks, masks = self.cuda(*items)

            # inpaint model
            if model == 2:
                # eval
                landmarks[landmarks >= self.config.INPUT_SIZE - 1] = self.config.INPUT_SIZE - 1
                landmarks[landmarks < 0] = 0

                landmark_map = torch.zeros(
                    (landmarks.shape[0], 1, self.config.INPUT_SIZE, self.config.INPUT_SIZE)).cuda()
                for i in range(landmarks.shape[0]):
                    landmark_map[i, 0, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i,0:self.config.LANDMARK_POINTS,0]] = 1

                inputs = (images * (1 - masks))

                for i in range(inputs.shape[0]):
                    inputs[i, :, landmarks[i, 0:self.config.LANDMARK_POINTS, 1],
                    landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1

                outputs = self.inpaint_model(images, landmark_map, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):

        self.landmark_model.eval()
        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)
        cal_mean_nme = self.cal_mean_nme()

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            images, landmarks, masks = self.cuda(*items)
            index += 1
            # landmark prediction model
            if model == 1:

                landmarks[landmarks>=self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
                path = os.path.join(self.results_path, self.model_name)

                name = self.test_dataset.load_name(index-1)

                landmark_pred = self.landmark_model(images*(1-masks)+masks, masks).detach()
                landmark_pred = (landmark_pred).long().reshape(-1, self.config.LANDMARK_POINTS, 2)
                landmark_pred[landmark_pred>=self.config.INPUT_SIZE-1]=self.config.INPUT_SIZE-1
                images_masked = images*(1-masks)
                images_masked[0, :, landmark_pred[0, :, 1], landmark_pred[0, :, 0]] = 1
                images_output = self.postprocess(images_masked)

                create_dir(path)

                file_name = os.path.join(path, name[:-4] + '.png')

                imsave(images_output[0], file_name)

                nme = self.cal_nme(landmarks,landmark_pred)

                print(str.format(name+' NME:{}, MEAN_NME:{}', nme, cal_mean_nme(nme)))
            # inpaint model
            elif model == 2:
                landmarks[landmarks >= self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
                landmarks[landmarks < 0] = 0

                landmark_map = torch.zeros((landmarks.shape[0], 1, self.config.INPUT_SIZE, self.config.INPUT_SIZE)).cuda()
                for i in range(landmarks.shape[0]):
                    landmark_map[i, 0, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1

                inputs = (images * (1 - masks))
                for i in range(inputs.shape[0]):
                    inputs[i, :, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1

                outputs = self.inpaint_model(images, landmark_map, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_masked = os.path.join(self.results_path,self.model_name,'masked')
                path_result = os.path.join(self.results_path, self.model_name,'result')
                path_joint = os.path.join(self.results_path,self.model_name,'joint')
                path_landmark_mask = os.path.join(self.results_path, self.model_name, 'landmark_mask')
                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)
                create_dir(path_landmark_mask)
                landmark_mask_image = images * (1 - masks) + masks
                landmark_mask_image = (landmark_mask_image.squeeze().cpu().numpy().transpose(1,2,0)*255).astype('uint8')
                landmark_mask_image = landmark_mask_image.copy()
                for i in range(landmarks.shape[1]):
                    circle(landmark_mask_image, (landmarks[0, i, 0], landmarks[0, i, 1]), radius=2,
                           color=(0, 255, 0), thickness=-1)


                masked_images = self.postprocess(images*(1-masks)+masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint,name[:-4]+'.png'))
                landmark_mask_image = Image.fromarray(landmark_mask_image)
                landmark_mask_image.save(os.path.join(path_landmark_mask, name))
                images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                imsave(masked_images,os.path.join(path_masked,name))
                imsave(images_result,os.path.join(path_result,name))

                print(name + ' complete!')

            # inpaint with joint model
            elif model == 3:
                output_landmark = self.landmark_model(images * (1 - masks) + masks,  masks)

                landmark_pred = (output_landmark.detach()).reshape(-1, self.config.LANDMARK_POINTS, 2).long()
                landmark_pred[landmark_pred >= self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
                landmark_last_map = self.generate_landmark_map(landmark_pred, self.config.INPUT_SIZE)

                images_out = self.inpaint_model(images, landmark_last_map, masks)

                outputs_merged = (images_out * masks) + (images * (1 - masks))

                landmark_image = images*(1-masks)
                landmark_image[0, :, landmark_pred[0, :, 1], landmark_pred[0, :, 0]] = 1

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(images*(1-masks)),
                    self.postprocess(landmark_image),
                    self.postprocess(images_out),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )


                path_masked = os.path.join(self.results_path, self.model_name, 'masked')
                path_result = os.path.join(self.results_path, self.model_name, 'result')
                path_joint = os.path.join(self.results_path, self.model_name, 'joint')
                path_landmark = os.path.join(self.results_path, self.model_name, 'landmark')
                path_landmark_mask = os.path.join(self.results_path, self.model_name, 'landmark_mask')
                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)
                create_dir(path_landmark)
                create_dir(path_landmark_mask)
                pure_landmark_image = torch.zeros(1, 3, images.shape[2], images.shape[3])
                pure_landmark_image = (pure_landmark_image[0].cpu().numpy().copy().transpose(1, 2, 0) * 255).astype('uint8')
                pure_landmark_image = pure_landmark_image.copy()


                landmark_mask_image = images*(1-masks)+masks
                landmark_mask_image = (landmark_mask_image[0].cpu().numpy().copy().transpose(1,2,0)*255).astype('uint8')
                landmark_mask_image = landmark_mask_image.copy()
                for i in range(landmark_pred.shape[1]):
                    circle(landmark_mask_image,(landmark_pred[0,i,0],landmark_pred[0,i,1]),radius=2,color=(0,255,0),thickness=-1)
                    circle(pure_landmark_image, (landmark_pred[0, i, 0], landmark_pred[0, i, 1]), radius=2,
                           color=(0, 255, 0), thickness=-1)
                masked_images = self.postprocess(images * (1 - masks) + masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint, name[:-4] + '.png'))
                landmark_mask_image = Image.fromarray(landmark_mask_image)
                landmark_mask_image.save(os.path.join(path_landmark_mask, name))
                pure_landmark_image = Image.fromarray(pure_landmark_image)
                pure_landmark_image.save(os.path.join(path_landmark, name))
                images_joint.save(os.path.join(path_joint, name[:-4] + '.png'))
                imsave(masked_images, os.path.join(path_masked, name))
                imsave(images_result, os.path.join(path_result, name))

        print('\nEnd Testing')



    def sample(self, it=None):
        self.inpaint_model.eval()
        self.landmark_model.eval()

        model = self.config.MODEL

        items = next(self.sample_iterator)

        if model == 1 and self.config.AUGMENTATION_TRAIN == 1:
            images, landmarks, masks, masks2, images_orig, edges_orig= self.cuda(*items)
        else:
            images,landmarks,masks = self.cuda(*items)

        landmarks[landmarks>=self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
        landmarks[landmarks<0] = 0

        # landmark model
        if model == 1:
            iteration = self.landmark_model.iteration
            masked_inputs = (images * (1 - masks))

            landmarks_pred = (self.landmark_model(images*(1-masks)+masks, masks)).reshape((-1,self.config.LANDMARK_POINTS,2)).long()
            landmarks_pred[landmarks_pred>=self.config.INPUT_SIZE] = self.config.INPUT_SIZE - 1
            landmarks_pred[landmarks_pred<0] = 0

            result_img = masked_inputs.clone()
            gt_masked_img = masked_inputs.clone()
            for i in range(landmarks_pred.shape[0]):
                result_img[i,:,landmarks_pred[i,:,1],landmarks_pred[i,:,0]] = 1
                gt_masked_img[i,:,landmarks[i,:,1],landmarks[i,:,0]] = 1

        # inpaint model
        elif model == 2:
            landmark_map = torch.zeros((landmarks.shape[0], 1, self.config.INPUT_SIZE, self.config.INPUT_SIZE)).cuda()
            for i in range(landmarks.shape[0]):
                landmark_map[i, 0, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            for i in range(inputs.shape[0]):
                inputs[i, :, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1-masks[i,0,landmarks[i, :, 1], landmarks[i,:,0]]

            outputs = self.inpaint_model(images, landmark_map, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))


        if it is not None:
            iteration = it


        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        if model == 1:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(masked_inputs),
                self.postprocess(gt_masked_img),
                self.postprocess(result_img),
                img_per_row = image_per_row
            )
        elif model == 2:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row=image_per_row
            )



        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generate_landmark_map(self, landmark_cord, img_size = 256):
        '''
        :param landmark_cord: [B,self.config.LANDMARK_POINTS,2] or [self.config.LANDMARK_POINTS,2], tensor or numpy array
        :param img_size:
        :return: landmark_img [B, 1, img_size, img_size] or [1, img_size, img_size], tensor or numpy array
        '''

        if torch.is_tensor(landmark_cord):
            if landmark_cord.ndimension() == 3:
                landmark_img = torch.zeros(landmark_cord.shape[0],1,img_size, img_size).cuda()
                for i in range(landmark_cord.shape[0]):
                    landmark_img[i,0,landmark_cord[i,:,1],landmark_cord[i,:,0]] = 1
            elif landmark_cord.ndimension() == 2:
                landmark_img = torch.zeros(1,img_size,img_size)
                landmark_img[0,landmark_cord[:,1],landmark_cord[:,0]] = 1

        return landmark_img

    def cal_nme(self,landmarks_gt, landmarks_pred):

        landmarks_gt = landmarks_gt.float()
        landmarks_pred = landmarks_pred.float()
        if self.config.LANDMARK_POINTS == 68:
            point1 = landmarks_gt[0, 36, :]
            point2 = landmarks_pred[0, 45, :]
        elif self.config.LANDMARK_POINTS == 98:
            point1 = landmarks_gt[0, 60, :]
            point2 = landmarks_pred[0, 72, :]
        distance = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        sum = 0
        for i in range(self.config.LANDMARK_POINTS):
            sum += torch.norm((landmarks_pred-landmarks_gt)[0,i,:]).item()
        sum = sum/(distance*self.config.LANDMARK_POINTS)
        return sum

    class cal_mean_nme():
        sum = 0
        amount = 0
        mean_nme = 0

        def __call__(self, nme):
            self.sum += nme
            self.amount += 1
            self.mean_nme = self.sum / self.amount
            return self.mean_nme

        def get_mean_nme(self):
            return self.mean_nme

