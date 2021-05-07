
import torch
import torch.nn.functional as F
import torch.nn as nn

from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from sample import GeneratedSample
import params
import os

class TimestampRegressionModel():

    def __init__(self):
        self.device = params.DEVICE
        self.loss_names = ['regression_loss']

        self.list_R = []
        self.model_names = ['R']

        self.netR = None # TODO load resnet 50 or 101 or something
        self.loss_function = None # TODO define loss function based on cosine similarity of angles for time on 24 hour scale
        self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=params.LR) # TODO what are betas for resnet?

        self.optimizers = [self.optimizer_R]
        self.scaler = GradScaler() # for mixed precision training; faster

    def set_input(self, input_real):
        self.real_images = input_real['image'].to(self.device)
        self.paths = input_real['path']
        
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

    def optimize_parameters(self):
        # TODO run forward pass
        # TODO run backward pass
        # TODO update weights
        self.scaler.update()
        