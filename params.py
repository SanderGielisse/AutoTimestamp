

import torch
import torch.nn.functional as F
import torch.nn as nn

# we define our settings here
DEVICE = torch.device('cuda')
DATA_DIR = "/media/ultra_ssd/TUDelft/deeplearningseminar/mirflickr"
EXIF_DIR = DATA_DIR + "/meta/exif/"
SAVE_DIR = "./checkpoints/"
PHASE = 'train'
IMAGE_SIZE = 256
BATCH_SIZE = 128
EPOCHS = 200 # TODO idk how large dataset is exactly?
LR = 1e-6 # TODO experimentally find good value
USE_AUTOCAST = True
SPLIT_RATIO = 0.95 # 80% for training
