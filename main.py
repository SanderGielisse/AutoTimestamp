# partly from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
# see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/LICENSE for license

import torch
import torch.nn.functional as F
import torch.nn as nn

import params
import time
import random
from models import TimestampRegressionModel
from visualizer import Visualizer
from image_dataset import ImageDataset
import image_dataset as id

class ImageDatasetDataLoader():

    def __init__(self, dataset):
        self.dataset = dataset
        print("Initialized dataset %s" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=8)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data

if __name__ == '__main__':
    total_iters = 0
    dataset_train, dataset_test = id.build_split()
    
    dataset_train = ImageDatasetDataLoader(dataset_train)
    dataset_test = ImageDatasetDataLoader(dataset_test)

    dataset_size = len(dataset_train)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = TimestampRegressionModel()
    visualizer = Visualizer()

    for epoch in range(params.EPOCHS):
        print("Running epoch %d..." % epoch)
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration

        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        """
        correct_quarter = 0
        correct_std = 0
        total = 0
        for i, input_real in enumerate(dataset_test):  # inner loop within one epoch
            model.set_input(input_real)         # unpack data from dataset and apply preprocessing
            l_correct_quarter, l_correct_std, l_total = model.test()   # calculate loss functions, get gradients, update network weights
            correct_quarter += l_correct_quarter
            correct_std += l_correct_std
            total += l_total
        print('QUARTER correct', correct_quarter, 'total', total, 'percentage', (correct_quarter * 1.0 / total))
        print('STD correct', correct_std, 'total', total, 'percentage', (correct_std * 1.0 / total))
        """

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        print('=== Running Epoch with lr', get_lr(model.optimizer_R), ' ===')

        for i, input_real in enumerate(dataset_train):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % 2048 == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += params.BATCH_SIZE
            epoch_iter += params.BATCH_SIZE
            model.set_input(input_real)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % 2048 == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / params.BATCH_SIZE
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % 20000 == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                # model.save_networks(save_suffix)

            iter_data_time = time.time()
        model.scheduler.step() # update learning rate
        print("Updating learning rate...")

        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        # model.save_networks('latest')
        # model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, params.EPOCHS, time.time() - epoch_start_time))
