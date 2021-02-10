import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from utils import *
from random import random



class ConvModule(nn.Module):
    def __init__(self, in_features, out_features, stride, kernel_size=3, padding=1, nonlinearity=torch.relu):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.bn = nn.BatchNorm2d(out_features)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        h = self.conv(x)
        #h = self.bn(h)  
        h = self.nonlinearity(h)
        return h

class FCModule(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=torch.relu):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        #self.bn = nn.BatchNorm1d(out_features)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        h = self.fc(x)
        #h = self.bn(h)
        h = self.nonlinearity(h)
        return h

class LowFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(1, 64, 2)
        self.conv2 = ConvModule(64, 128, 1)

        self.conv3 = ConvModule(128, 128, 2)
        self.conv4 = ConvModule(128, 256, 1)

        self.conv5 = ConvModule(256, 256, 2)
        self.conv6 = ConvModule(256, 512, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        o = self.conv6(h)
        return o

class MidFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(512, 512, 1)
        self.conv2 = ConvModule(512, 256, 1)

    def forward(self, x):
        h = self.conv1(x)
        o = self.conv2(h)
        return o

class GlobalFeatures(nn.Module):
    """Return the output of the last layer but also the output of the second-to-last layer,
    For the classifier""" 
    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(512, 512, 2)
        self.conv2 = ConvModule(512, 512, 1)

        self.conv3 = ConvModule(512, 512, 2)
        self.conv4 = ConvModule(512, 512, 1)

        self.fc1 = FCModule(7 * 7 * 512, 1024)
        self.fc2 = FCModule(1024, 512)
        self.fc3 = FCModule(512, 256)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = h.view(-1,7*7*512)
        h = self.fc1(h)
        h = self.fc2(h)
        o = self.fc3(h)
        return o, h

class Colorizer(nn.Module):
    """Input the concatenation of mid level and global level features"""
    def __init__(self):
        super().__init__()
        self.fusion = ConvModule(512, 256, 1, kernel_size=1, padding=0)
        self.conv1 = ConvModule(256, 128, 1)

        self.us1 = nn.Upsample(scale_factor=2)
        self.conv2 = ConvModule(128, 64, 1)
        self.conv3 = ConvModule(64, 64, 1)

        self.us2 = nn.Upsample(scale_factor=2)
        self.conv4 = ConvModule(64, 32, 1)
        self.conv5 = ConvModule(32, 2, 1, nonlinearity=torch.sigmoid)

        self.us3 = nn.Upsample(scale_factor=2)


    def forward(self, x):
        h = self.fusion(x)
        h = self.conv1(h)
        #h = self.us1(h)
        h = nn.functional.interpolate(input=h, scale_factor=2)

        h = self.conv2(h)
        h = self.conv3(h)
        #h = self.us2(h)

        h = nn.functional.interpolate(input=h, scale_factor=2)
        h = self.conv4(h)
        h = self.conv5(h)
        #o = self.us3(h)

        o = nn.functional.interpolate(input=h, scale_factor=2)

        return o

class Classifier(nn.Module):
    def __init__(self, nclasses=365): #365 or 285 ?
        super().__init__()
        self.fc1 = FCModule(512, 400)
        self.fc2 = FCModule(400, nclasses,nonlinearity=nn.Identity())

    def forward(self, x):
        h = self.fc1(x)
        o = self.fc2(h)
        #o = torch.softmax(h, dim=1) 
        return o

class LTBC(pl.LightningModule):
    def __init__(self, alpha, lr=0.0001, rightsize=False, classify=True):
        super().__init__()
        self.save_hyperparameters()

        self.rightsize = rightsize
        self.classify = classify
        self.alpha = alpha
        self.lr = lr

        self.low = LowFeatures()
        self.mid = MidFeatures()
        self.glob = GlobalFeatures()
        self.color = Colorizer()
        if self.classify:
            self.classif = Classifier()

        self.crit_image = nn.MSELoss(reduction='mean')
        self.crit_label = nn.CrossEntropyLoss()

    def forward(self, inputs):
        low_feats_1 = self.low(inputs)
        if self.rightsize:
            low_feats_2 = low_feats_1
        else:
            pass # TODO
        mid_feats = self.mid(low_feats_1)
        glob_feats, hidden_activ = self.glob(low_feats_2)
        glob_feats = glob_feats[:, :, None, None].expand(-1, -1, mid_feats.shape[2], mid_feats.shape[3])

        feats = torch.cat([mid_feats, glob_feats], dim=1)
        ab = self.color(feats)
        if self.classify:
            label = self.classif(hidden_activ)
        else:
          label = None
        return ab, label

    def configure_optimizers(self):
        # No further info about optimizer was provided in the paper, we only know it was adadelta
        #optim = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def training_step(self, batch, batch_idx):
        image, label = batch
        L_image = image[:, :1, :, :]
        ab_image = image[:, 1:, :, :]
        pred_ab, pred_label = self(L_image)
        loss = self.crit_image(ab_image, pred_ab)
        self.log('train_color_loss', loss,prog_bar=False)

        if self.classify:
            classif_loss = self.alpha * self.crit_label(pred_label, label)
            loss += classif_loss
            self.log('train_classif_loss', classif_loss, prog_bar=False)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        L_image = image[:, :1, :, :]
        ab_image = image[:, 1:, :, :]
        pred_ab, pred_label = self(L_image)
        loss = self.crit_image(ab_image, pred_ab)
        self.log('val_color_loss', loss,prog_bar=True)
        if self.classify:
            classif_loss = self.alpha * self.crit_label(pred_label, label)
            loss += classif_loss
            self.log('val_classif_loss', classif_loss, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        self.log_images(L_image, ab_image, pred_ab)
        return loss

    def log_images(self, L_image, ab_image, ab_pred):
        for img_idx in range(L_image.shape[0]):
            # Log one image out of 10
            if random() > 0.9:
                L_image = L_image.detach().cpu()
                ab_image = ab_image.detach().cpu()
                ab_pred = ab_pred.detach().cpu()
                gt_Lab_image = torch.cat([L_image, ab_image], dim=1)
                gt_rgb_image = convert_back_to_rgb(gt_Lab_image.detach()[img_idx,:1,:,:], gt_Lab_image.detach()[img_idx,1:,:,:])

                pred_Lab_image = torch.cat([L_image, ab_pred], dim=1)
                pred_rgb_image = convert_back_to_rgb(pred_Lab_image.detach()[img_idx,:1,:,:], pred_Lab_image.detach()[img_idx,1:,:,:])
                
                three_images= torch.stack([torch.tensor(gt_rgb_image),
                                        torch.tensor(pred_rgb_image),
                                        torch.tensor(color.rgb2gray(pred_rgb_image)).unsqueeze(2).expand(-1, -1, 3)])
                side_by_side = torchvision.utils.make_grid(three_images.permute(0, 3, 1, 2))
                self.logger.experiment.add_image(f'comparison: image {img_idx}', side_by_side, 0)

