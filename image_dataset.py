import os.path
import torchvision.transforms as transforms
from PIL import Image
from ast import literal_eval
import numpy as np
import torch
import torchvision
import random
import torch.utils.data as data
from PIL import Image
from abc import ABC, abstractmethod

class ImageDataset(data.Dataset, ABC):

    def __init__(self, data_dir, phase, batch_size):
        self.dir_AB = os.path.join(data_dir, phase)  # get the image directory
        self.all_paths = sorted(make_dataset(self.dir_AB))  # get image paths
        random.shuffle(self.all_paths)
        self.batch_size = batch_size

    def __getitem__(self, index):
        # read an image given a random integer index
        image_path = self.all_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        if random.choice([True, False]) :
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        tr = transforms.Compose([transforms.ToTensor()])
        image = tr(image)
        image = image * 2.0 - 1.0
        return {'image': image, 'path': image_path}

    def __len__(self):
        batches = len(self.all_paths) // self.batch_size
        return batches * self.batch_size


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.webp'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
