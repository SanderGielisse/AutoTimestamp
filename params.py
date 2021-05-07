

import torch
import torch.nn.functional as F
import torch.nn as nn

# we define our settings here
DEVICE = torch.device('cuda')
DATA_DIR = "/media/ultra_ssd/lsun/dataset_bedrooms/"
SAVE_DIR = "./checkpoints/"
PHASE = 'train'
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10 # TODO idk how large dataset is exactly?
LR = 2e-4 # TODO experimentally find good value
USE_AUTOCAST = True
