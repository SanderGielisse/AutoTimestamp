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
import params
import math

ENTRY_NAME = "-Date and Time (Original)"

def build_split():
    image_paths = make_dataset(params.DATA_DIR)
    time_stamps = []
    
    def load(id):
        # parse -Date and Time (Original)
        # 2008:06:21 16:12:37
        with open(params.EXIF_DIR + 'exif' + str(id) + ".txt", 'r', errors='replace') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
            # print(lines)
            if ENTRY_NAME not in lines:
                return None
            i = lines.index(ENTRY_NAME)
            l = lines[i + 1]
            l_split = l.split(" ")
            if len(l_split) != 2:
                return None
            value = l_split[1]
            if value.strip() == "":
                return None
            # print(value)
            split = value.split(":")
            if len(split) != 3:
                return None
            hour = float(split[0])
            minute = float(split[1])
            second = float(split[2])
            result = hour + (minute / 60.0) + (second / (60.0 * 60.0))
            return (result / 12.0) * math.pi - math.pi

    for image in image_paths:
        # extract the name
        name = image.split("/")[-1]
        name = name.replace("im", "")
        name = name.replace(".jpg", "")
        time = load(name)
        if time is None:
            continue
        time_stamps.append(time)

    pairs = list(zip(image_paths, time_stamps))
    random.shuffle(pairs)

    split_at = int(len(pairs) * params.SPLIT_RATIO)
    train = pairs[:split_at]
    test = pairs[split_at:]

    return ImageDataset(train), ImageDataset(test)

class ImageDataset(data.Dataset):

    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
        random.shuffle(self.data_pairs)

    def __getitem__(self, index):
        # read an image given a random integer index
        image_path, y = self.data_pairs[index]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((params.IMAGE_SIZE, params.IMAGE_SIZE))
        if random.choice([True, False]) :
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        tr = transforms.Compose([transforms.ToTensor()])
        image = tr(image)
        image = image * 2.0 - 1.0

        res = torch.zeros((1,))
        res[0] = y

        return {'image': image, 'y': res}

    def __len__(self):
        batches = len(self.data_pairs) // params.BATCH_SIZE
        return batches * params.BATCH_SIZE


IMG_EXTENSIONS = [
    '.jpg'
]

""" if needed can add
, '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.webp'
"""

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
