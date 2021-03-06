
import torch
import torch.nn.functional as F
import torch.nn as nn

# we define our settings here
DEVICE = torch.device('cuda')
DATA_DIR = "/media/ultra_ssd/TUDelft/deeplearningseminar/flickrdataset10000/images/"
EXIF_DIR = DATA_DIR + "../images_meta/"
SAVE_DIR = "./checkpoints/"
PHASE = 'train'
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100 # TODO idk how large dataset is exactly?
LR = 1e-5 # TODO experimentally find good value
USE_AUTOCAST = True
SPLIT_RATIO = 0.9 # N% for training
