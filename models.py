
import torch
import torch.nn.functional as F
import torch.nn as nn

import math
from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import params
import os
import resnet as rn
import loss_function as vm
from torch.distributions.von_mises import VonMises
from torchsummary_local import summary
import torchvision
from scipy import misc

import matplotlib.pyplot as plt
from PIL import Image

class TimestampRegressionModel():

    def __init__(self):
        self.device = params.DEVICE
        self.loss_names = ['loss_train', 'loss_test']

        self.list_loss_train = []
        self.list_loss_test = []

        self.model_names = ['R']

        resnet = torchvision.models.resnet18(pretrained=True)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        # for param in resnet.parameters():
        #     param.requires_grad = False

        layers = []
        layers += [ resnet, nn.Flatten() ]
        layers += [ nn.Linear(512, 2) ]
        resnet = nn.Sequential(*layers)

        self.netR = resnet.to(self.device)
        sah_summary(self.netR, (3, 224, 224))

        self.loss_function = vm.compute_loss_regression # vm.compute_loss
        self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=params.LR) #, momentum=0.9, weight_decay=0.0001) # 
        # self.scheduler = StepLR(self.optimizer_R, step_size=1, gamma=0.1, verbose=True) # does nothing
        self.val_loss = None

        self.optimizers = [self.optimizer_R]
        self.scaler = GradScaler() # for mixed precision training; faster

        self.tanh = nn.Tanh()
        # self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        # self.sigmoid = nn.Sigmoid()
        # self.elu = nn.ELU()
        # self.tanh = nn.Hardtanh(min_val = -1.0 - 1e-6, max_val = 1.0 + 1e-6)

    def set_input(self, input_real):
        # 'image': image, 'y'
        self.images = input_real['image'].to(self.device)
        self.ys = input_real['y'].to(self.device)
        self.imgs01 = input_real['img01']# .to(self.device)

        if torch.isnan(self.images).any():
            raise Exception('images had NaN')
        if torch.isnan(self.ys).any():
            raise Exception('ys had NaN')
        if torch.isnan(self.imgs01).any():
            raise Exception('imgs01 had NaN')
        
    def save_networks(self, name):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (name, name)
                save_path = os.path.join(params.SAVE_DIR, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda()
                # torch.save(net.cpu().state_dict(), save_path)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                list = getattr(self, 'list_' + name)
                if len(list) == 0:
                    value = 0
                else:
                    value = sum(list) / len(list)
                    list.clear()
                errors_ret[name] = value
        return errors_ret

    def save(self, i, img01, pred, actual, loss):

        def tohours(v):
            v /= math.pi
            v *= 12.0
            return v

        pred = tohours(pred)
        actual = tohours(actual)
        loss = tohours(loss)

        img01 = (img01 * 255).permute(1, 2, 0).int()

        # im = Image.fromarray(img01)
        plt.imshow(img01)
        plt.title("pred[%f] actual[%f] loss[%f]" % (pred, actual, loss))
        plt.savefig('./test_dumps/' + str(i) + '.png')
        plt.close()

    def test(self, save):
        with torch.no_grad():
            with autocast():
                x = self.images
                x = self.netR(x)
                if torch.isnan(x).any():
                    raise Exception('network output NaN', x)
                mus_x = self.tanh(x[:, 0])
                mus_y = self.tanh(x[:, 1])
                ys = self.ys[:, 0]
                mus = torch.atan2(mus_y, mus_x) # angle with (1,0) -> between -pi and pi
                losses = self.loss_function(mus, self.ys[:, 0])

                if save:
                    for i in range(x.shape[0]):
                        img01 = self.imgs01[i]
                        pred = float(mus[i])
                        actual = float(ys[i])
                        loss = float(losses[i])
                        self.save(i, img01, pred, actual, loss)

                return torch.mean(losses)


    def optimize_parameters(self):

        def to_hours(v):
            # angle is given between 0 and pi (0 to 12 hours off)
            v /= math.pi
            v *= 12.0
            return v

        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        self.netR.zero_grad()

        with autocast():
            x = self.images
            x = self.netR(x)
            if torch.isnan(x).any():
                raise Exception('network output NaN', x)

            mus_x = self.tanh(x[:, 0])
            mus_y = self.tanh(x[:, 1])
            ys = self.ys[:, 0]
            mus = torch.atan2(mus_y, mus_x) # angle with (1,0) -> between -pi and pi
            loss = torch.mean(self.loss_function(mus, self.ys[:, 0]))

        self.list_loss_train.append(to_hours(float(loss)))
        self.list_loss_test.append(to_hours(float(self.val_loss)))

        self.scaler.scale(loss).backward() # 
        self.scaler.step(self.optimizer_R) # self.scaler.
        self.scaler.update()


def sah_summary(net, shape):
    if not torch.cuda.is_available():
        return None
    model = net.to("cuda")
    summary(model, shape)
    return None
