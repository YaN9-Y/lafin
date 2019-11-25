import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, TVLoss



class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else: 
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + landmark(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=4, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        self.tv_loss = TVLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )



    def process(self, images, landmarks, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs

        outputs = self(images, landmarks, masks)

        
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(torch.cat((dis_input_real, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_fake, _ = self.discriminator(torch.cat((dis_input_fake, landmarks), dim=1))                   # in: [rgb(3)+landmark(1)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(torch.cat((gen_input_fake, landmarks), dim=1))                   # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        #generator tv loss
        tv_loss = self.tv_loss(outputs*masks+images*(1-masks))
        gen_loss += self.config.TV_LOSS_WEIGHT * tv_loss

        # create logs
        logs = [
            ("gLoss",gen_loss.item()),
            ("dLoss",dis_loss.item())
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, landmarks, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, landmarks), dim=1)
        scaled_masks_quarter = F.interpolate(masks, size=[int(masks.shape[2] / 4), int(masks.shape[3] / 4)],
                                     mode='bilinear', align_corners=True)
        scaled_masks_half = F.interpolate(masks, size=[int(masks.shape[2] / 2), int(masks.shape[3] / 2)],
                                     mode='bilinear', align_corners=True)

        outputs = self.generator(inputs,masks,scaled_masks_half,scaled_masks_quarter)                                    # in: [rgb(3) + landmark(1)]
        return outputs

    def backward(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

    def backward_joint(self, gen_loss = None, dis_loss = None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


from .networks import MobileNetV2

def abs_smooth(x):
    absx = torch.abs(x)
    minx = torch.min(absx,other=torch.ones(absx.shape).cuda())
    r = 0.5 *((absx-1)*minx + absx)
    return r

def loss_landmark_abs(y_true, y_pred):
    loss = torch.mean(abs_smooth(y_pred - y_true))
    return loss

def loss_landmark(landmark_true, landmark_pred, points_num=68):
    landmark_loss = torch.norm((landmark_true-landmark_pred).reshape(-1,points_num*2),2,dim=1,keepdim=True)

    return torch.mean(landmark_loss)

class LandmarkDetectorModel(nn.Module):
    def __init__(self, config):
        super(LandmarkDetectorModel, self).__init__()
        self.mbnet = MobileNetV2(points_num=config.LANDMARK_POINTS)
        self.name = 'landmark_detector'
        self.iteration = 0
        self.config = config

        self.landmark_weights_path = os.path.join(config.PATH, self.name + '.pth')

        if len(config.GPU) > 1:
            self.mbnet = nn.DataParallel(self.mbnet, config.GPU)

        self.optimizer = optim.Adam(
            params=self.mbnet.parameters(),
            lr=self.config.LR,
            weight_decay=0.000001
        )


    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'detector': self.mbnet.state_dict()
        }, self.landmark_weights_path)

    def load(self):
        if os.path.exists(self.landmark_weights_path):
            print('Loading landmark detector...')

            if torch.cuda.is_available():
                data = torch.load(self.landmark_weights_path)
            else:
                data = torch.load(self.landmark_weights_path, map_location=lambda storage, loc: storage)

            self.mbnet.load_state_dict(data['detector'])
            self.iteration = data['iteration']
            print('Loading landmark detector complete!')

    def forward(self, images, masks):
        images_masked = images* (1 - masks).float() + masks

        landmark_gen = self.mbnet(images_masked)
        landmark_gen *= self.config.INPUT_SIZE

        return landmark_gen

    def process(self, images, masks, landmark_gt):
        self.iteration += 1
        self.optimizer.zero_grad()

        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape((-1, self.config.LANDMARK_POINTS, 2))
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss", loss.item())]
        return landmark_gen, loss, logs

    def process_aug(self, images, masks, landmark_gt):
        self.optimizer.zero_grad()
        images_masked = images*(1-masks)+masks
        landmark_gen = self(images_masked, masks)
        landmark_gen = landmark_gen.reshape(-1,self.config.LANDMARK_POINTS,2)
        loss = loss_landmark(landmark_gt.float(),landmark_gen, points_num=self.config.LANDMARK_POINTS)

        logs = [("loss_aug", loss.item())]

        return landmark_gen, loss, logs



    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
