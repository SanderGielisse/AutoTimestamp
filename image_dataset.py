import os.path
import torchvision.transforms as transforms
from PIL import Image
from ast import literal_eval
import numpy as np
import torch
import torchvision
import random
import torch.utils.data as data
from PIL import Image #, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import params
import math
import tqdm
import pickle

ENTRY_NAME = "-Date and Time (Original)"

def build_split():
    image_paths_all = make_dataset(params.DATA_DIR)
    # format /media/ultra_ssd/TUDelft/deeplearningseminar/mirflickr_full/images/0/0.jpg

    """
    # load the valid paths from data_times.pickle
    with open('./data_times.pickle', 'rb') as f:
        valid_pairs = pickle.load(f)
    valid_paths = []
    for path, _ in valid_pairs:
        valid_paths.append(path)
    """
    valid_paths = image_paths_all

    def to_pairs(selected_paths):
        def load(folder, id):
            # parse -Date and Time (Original)
            # 2008:06:21 16:12:37
            with open(params.EXIF_DIR + folder + "/" + id + ".txt", 'r', errors='replace') as f:
                lines = f.readlines()
                lines = [x.strip() for x in lines]
                # print(lines)
                """
                if ENTRY_NAME not in lines:
                    print('skipping, not date time')
                    return None
                i = lines.index(ENTRY_NAME)
                l = lines[i + 1]
                """
                l = lines[0]
                l_split = l.split(" ")
                if len(l_split) != 2:
                    print('skipping, length != 2')
                    return None
                value = l_split[1]
                if not value.replace(":", "").isdecimal():
                    print('skipping, not decimal')
                    return None
                if value.strip() == "":
                    print('skipping, strip empty')
                    return None
                # print(value)
                split = value.split(":")
                if len(split) != 3:
                    print('skipping, length != 3')
                    return None
                # print(value)
                hour = int(split[0])
                if hour < 0 or hour > 23:
                    print('skipping, hour > 23')
                    return None
                minute = int(split[1])
                if minute < 0 or minute > 59:
                    print('skipping, minute > 59')
                    return None
                second = int(split[2])
                if second < 0 or second > 59:
                    print('skipping, second > 59')
                    return None
                result = hour + (minute / 60.0) + (second / (60.0 * 60.0))
                res = (result / 24.0) * 2 * math.pi # [0, 2pi]
                if res < 0 or res > 2 * math.pi:
                    raise Exception('res out of bounds ', res)
                return res, hour

        # time_stamps = []
        # image_paths = []
        buckets = {}
        
        for image in tqdm.tqdm(selected_paths): # or all with the check below
            # extract the name
            """
            try:
                Image.open(image).convert('RGB')
            except:
                print("SKIPPING ", image)
                continue
            """

            split = image.split("/")
            folder = "" # split[-2]
            name = split[-1]
            name = name.replace(".jpg", "")
            r = load(folder, name)
            if r is None:
                continue
            full_time, hour = r

            if full_time < 0 or full_time > 2 * math.pi:
                raise Exception('time out of bounds ', res)
            pair = (image, full_time)

            if hour not in buckets:
                buckets[hour] = []
            buckets[hour].append(pair)

        print('buckets', buckets)

        # now convert to evenly sized buckets via oversampling
        largest = 0
        for lis in buckets.values():
            l = len(lis)
            if l > largest:
                largest = l
        print('oversampling all buckets to size', largest)

        pairs = []
        for lis in buckets.values():
            pool = []
            for _ in range(largest):
                if len(pool) == 0:
                    pool = list(lis)
                    random.shuffle(pool)
                pairs += [pool.pop()]
            print('pairs size now', len(pairs))
        return pairs

    random.shuffle(valid_paths)
    # pairs = pairs[0:1000] # for quick testing

    split_at = int(len(valid_paths) * params.SPLIT_RATIO)
    train = valid_paths[:split_at]
    test = valid_paths[split_at:]

    print('train size before', len(train))
    print('test size before', len(test))

    train = to_pairs(train)
    test = to_pairs(test)

    print('train size after', len(train))
    print('test size after', len(test))

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

        img01 = transforms.Compose([transforms.ToTensor()])(image)

        if random.choice([True, False]) :
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        tr = transforms.Compose([transforms.RandomRotation(45), transforms.ToTensor(), normalize])
        image = tr(image)
        # image = image * 2.0 - 1.0

        # random 224x224 center crop
        max_offset = 256 - params.IMAGE_SIZE
        x_off = random.randrange(max_offset)
        y_off = random.randrange(max_offset)
        image = image[:, x_off:x_off+params.IMAGE_SIZE, y_off:y_off+params.IMAGE_SIZE] # C,H,W
        # print('size', image.shape)

        res = torch.zeros((1,))
        res[0] = y
        
        if y < 0 or y > 2 * math.pi:
            raise Exception('y out of bounds ', res)

        return {'image': image, 'y': res, 'img01': img01}

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
