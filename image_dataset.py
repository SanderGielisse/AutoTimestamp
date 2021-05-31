import os.path
import torchvision.transforms as transforms
from PIL import Image
from ast import literal_eval
import numpy as np
import torch
import torchvision
import random
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import params
import math
import tqdm
import pickle

ENTRY_NAME = "-Date and Time (Original)"

def build_split():
    image_paths = make_dataset(params.DATA_DIR)
    # format /media/ultra_ssd/TUDelft/deeplearningseminar/mirflickr_full/images/0/0.jpg

    pickle_path = "./data.pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            pairs = pickle.load(f)
    
    else:
        time_stamps = []
        buckets = np.zeros((24,))
        
        def load(folder, id):
            # parse -Date and Time (Original)
            # 2008:06:21 16:12:37
            with open(params.EXIF_DIR + folder + "/" + id + ".txt", 'r', errors='replace') as f:
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
                if not value.replace(":", "").isdecimal():
                    return None
                if value.strip() == "":
                    return None
                # print(value)
                split = value.split(":")
                if len(split) != 3:
                    return None
                # print(value)
                hour = int(split[0])
                if hour < 0 or hour > 23:
                    return None
                buckets[hour] += 1
                minute = int(split[1])
                second = int(split[2])
                result = hour + (minute / 60.0) + (second / (60.0 * 60.0))
                return (result / 12.0) * math.pi - math.pi # [-pi, pi]

        for image in tqdm.tqdm(image_paths):
            # extract the name

            try:
                Image.open(image).convert('RGB')
            except:
                continue
            
            split = image.split("/")
            folder = split[-2]
            name = split[-1]
            name = name.replace(".jpg", "")
            time = load(folder, name)
            if time is None:
                continue
            time_stamps.append(time)

        print('buckets', buckets)
        buckets /= np.sum(buckets)
        print('buckets', buckets)
        pairs = list(zip(image_paths, time_stamps))
        with open(pickle_path, 'wb') as f:
            pickle.dump(pairs, f)

    random.shuffle(pairs)
    # pairs = pairs[0:1000] # for quick testing

    split_at = int(len(pairs) * params.SPLIT_RATIO)
    train = pairs[:split_at]
    test = pairs[split_at:]

    print('train size', len(train))
    print('test size', len(test))

    return ImageDataset(train), ImageDataset(test)

class ImageDataset(data.Dataset):

    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
        random.shuffle(self.data_pairs)

    def __getitem__(self, index):
        # read an image given a random integer index
        image_path, y = self.data_pairs[index]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        if random.choice([True, False]) :
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        tr = transforms.Compose([transforms.ToTensor()])
        image = tr(image)
        image = image * 2.0 - 1.0

        # random 224x224 center crop
        max_offset = 256 - params.IMAGE_SIZE
        x_off = random.randrange(max_offset)
        y_off = random.randrange(max_offset)
        image = image[:, x_off:x_off+params.IMAGE_SIZE, y_off:y_off+params.IMAGE_SIZE] # C,H,W
        # print('size', image.shape)

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
