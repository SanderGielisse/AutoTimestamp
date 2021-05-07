
import torch
import torch.nn.functional as F
import torch.nn as nn

import params
import time
import random
from models import TimestampRegressionModel
from visualizer import Visualizer
from image_dataset import ImageDataset
from sample import GeneratedSample, Population


class ImageDatasetDataLoader():

    def __init__(self):
        self.dataset = ImageDataset(params.DATA_DIR, params.PHASE, params.BATCH_SIZE)
        print("Initialized dataset %s" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=8)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for _, data in enumerate(self.dataloader):
            yield data

if __name__ == '__main__':
    total_iters = 0
    dataset = ImageDatasetDataLoader()
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = TimestampRegressionModel()
    visualizer = Visualizer()

    for epoch in range(params.EPOCHS):
        print("Running epoch %d..." % epoch)
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration

        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, input_real in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % 100 == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += params.BATCH_SIZE
            epoch_iter += params.BATCH_SIZE
            model.set_input(input_real)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % 100 == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch, False)

            if total_iters % 100 == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / params.BATCH_SIZE
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % 20000 == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                # model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, params.EPOCHS, time.time() - epoch_start_time))
