
import torch
import torch.nn.functional as F
import torch.nn as nn

import math
from collections import OrderedDict
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import params
import os
import resnet as rn
import loss_function as vm
from torch.distributions.von_mises import VonMises
from torchsummary_local import summary

class TimestampRegressionModel():

    def __init__(self):
        self.device = params.DEVICE
        self.loss_names = ['loss_von_mises', 'loss_k_penalty', 'loss_total', 'average_k']

        self.list_loss_total = []
        self.list_loss_von_mises = []
        self.list_loss_k_penalty = []
        self.list_average_k = []

        self.model_names = ['R']

        resnet = rn.resnet18(pretrained=True, num_classes=1000)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        # sah_summary(resnet, (3, 224, 224))

        resnet = nn.Sequential(resnet, nn.Flatten(), nn.Linear(512, 2))

        self.netR = resnet.to(self.device)
        sah_summary(self.netR, (3, 224, 224))

        self.loss_function = vm.compute_loss_regression2 # vm.compute_loss
        self.optimizer_R = torch.optim.SGD(self.netR.parameters(), lr=params.LR, momentum=0.9, weight_decay=0.0001) # 
        self.scheduler = StepLR(self.optimizer_R, step_size=1, gamma=0.1)

        self.optimizers = [self.optimizer_R]
        # self.scaler = GradScaler() # for mixed precision training; faster

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def set_input(self, input_real):
        # 'image': image, 'y'
        self.images = input_real['image'].to(self.device)
        self.ys = input_real['y'].to(self.device)

        if torch.isnan(self.images).any():
            raise Exception('images had NaN')
        if torch.isnan(self.ys).any():
            raise Exception('ys had NaN')
        
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

    """
    def test(self):
        correct_quarter = 0
        correct_std = 0
        total = 0

        with torch.no_grad():
            x = self.images
            x = self.netR(x) # [N, 3]

            # mu_x -> 
            mus_x = self.tanh(x[:, 0])
            # mu_y -> 
            mus_y = self.tanh(x[:, 1])
            # k -> 
            # ks = self.elu(x[:, 2]) + 1 + 1e-6

            mus = vm.compute_mu_angle(mus_x, mus_y)

            ys = self.ys[:, 0]
            for i in range(ys.shape[0]):
                m = float(mus[i])
                # k = float(ks[i])
                k = 1
                y = float(ys[i])
                # diff = abs(m - y)
                # raw_diff = first > second ? first - second : second - first
                phi = abs(m - y) % (math.pi * 2)
                distance = (math.pi * 2) - phi if phi > math.pi else phi

                cor_quarter = distance < (math.pi / 4)
                
                vml = VonMises(m, k)
                std = vml.variance.sqrt()
                cor_std = distance < std

                # print('pred', m, 'actual', y, 'diff', distance, 'cor_quarter', cor_quarter, 'cor_std', cor_std)
                total += 1
                if cor_quarter:
                    correct_quarter += 1
                if cor_std:
                    correct_std += 1
        return correct_quarter, correct_std, total
    """

    def optimize_parameters(self):

        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            self.netR.zero_grad()

            # with autocast():
            x = self.images
            x = self.netR(x) # [N, 3]
            if torch.isnan(x).any():
                raise Exception('network output NaN', x)

            # mu_x -> 
            mus_x = x[:, 0]
            # mu_y -> 
            mus_y = x[:, 1]

            """
            if (mus_x > 1).any() or (mus_x < -1).any():
                raise Exception('tanh mus_x out of bounds')
            if (mus_y > 1).any() or (mus_y < -1).any():
                raise Exception('tanh mus_y out of bounds')
            """

            ys = self.ys[:, 0]
            print('mus_x', mus_x)
            print('mus_y', mus_y)
            print('actual_x', torch.cos(ys))
            print('actual_y', torch.sin(ys))

            # calculate vector length
            length = torch.sqrt(mus_x * mus_x + mus_y * mus_y)
            mus_x_res = mus_x / length
            mus_y_res = mus_y / length

            # k -> 
            # ks = self.elu(x[:, 2]) + 1 + 1e-6

            # mus = vm.compute_mu_angle(mus_x, mus_y)
            # ks = torch.ones(mus.shape, device=params.DEVICE)
            loss_von_mises = self.loss_function(mus_x_res, mus_y_res, self.ys[:, 0]) # as middle one ks, 
            # loss_k = (1.0 / torch.square(ks)).mean() * 0
            loss_k = 0.01 * torch.mean(torch.square(length - 1))
            loss = loss_von_mises + loss_k

            # self.list_average_k.append(float(ks.mean()))

            self.list_loss_total.append(float(loss))
            self.list_loss_von_mises.append(float(loss_von_mises))
            self.list_loss_k_penalty.append(float(loss_k))

            loss.backward() # self.scaler.scale
            self.optimizer_R.step() # self.scaler.
            # self.scaler.update()
        
def sah_summary(net, shape):
    if not torch.cuda.is_available():
        return None
    model = net.to("cuda")
    summary(model, shape)
    return None
