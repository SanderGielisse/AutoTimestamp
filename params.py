
import torch
import torch.nn.functional as F
import torch.nn as nn

# we define our settings here
DEVICE = torch.device('cuda')
DATA_DIR = "/media/ultra_ssd/TUDelft/deeplearningseminar/mirflickr_full/images/"
EXIF_DIR = DATA_DIR + "../exif/"
SAVE_DIR = "./checkpoints/"
PHASE = 'train'
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10 # TODO idk how large dataset is exactly?
LR = 1e-5 # TODO experimentally find good value
USE_AUTOCAST = True
SPLIT_RATIO = 0.99 # N% for training
