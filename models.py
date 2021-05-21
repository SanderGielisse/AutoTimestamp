
import torch
import torch.nn.functional as F
import torch.nn as nn

import math
from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
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

        self.netR = rn.resnet18(pretrained=False, num_classes=2).to(self.device)
        sah_summary(self.netR, (3, 256, 256))

        self.loss_function = vm.compute_loss
        self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=params.LR)

        self.optimizers = [self.optimizer_R]
        self.scaler = GradScaler() # for mixed precision training; faster

        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

    def set_input(self, input_real):
        # 'image': image, 'y'
        self.images = input_real['image'].to(self.device)
        self.ys = input_real['y'].to(self.device)
        
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

    def optimize_parameters(self):

        self.netR.zero_grad()

        with autocast():
            x = self.images
            x = self.netR(x) # [N, 3]
            # mu_x -> 
            mus_x = self.tanh(x[:, 0])
            # mu_y -> 
            mus_y = self.tanh(x[:, 1])
            # k -> 
            # ks = self.elu(x[:, 2]) + 1 + 1e-6

            mus = vm.compute_mu_angle(mus_x, mus_y)
            ks = torch.ones(mus.shape, device=params.DEVICE)
            loss_von_mises = self.loss_function(mus, ks, self.ys[:, 0])
            loss_k = (1.0 / torch.square(ks)).mean() * 0
            loss = loss_von_mises + loss_k

        self.list_average_k.append(float(ks.mean()))

        self.list_loss_total.append(float(loss))
        self.list_loss_von_mises.append(float(loss_von_mises))
        self.list_loss_k_penalty.append(float(loss_k))

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer_R)
        self.scaler.update()
        
def sah_summary(net, shape):
    if not torch.cuda.is_available():
        return None
    model = net.to("cuda")
    summary(model, shape)
    return None
