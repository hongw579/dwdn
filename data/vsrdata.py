
import os
import glob

from data import common

import numpy as np
import imageio
import random
import torch
import torch.utils.data as data


class VSRData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train

        self.n_image = []

        if train:
            self._set_filesystem(args.dir_data)

        else:
            self._set_filesystem(args.dir_data_test)

        self.images_sharp, self.images_blur, self.images_kernel = self._scan()
        self.num_image = np.load(self.images_sharp[0]).shape[0]
        print("Number of images to load:", self.num_image)

        if train:
            self.repeat = 1

        if args.process:
            self.data_sharp, self.data_blur, self.data_kernel = self._load() #'if True, load all dataset at once at RAM'

    def _scan(self):
        """
        Returns a list of image directories
        """
        image_names_blur = sorted(glob.glob(os.path.join(self.dir_image_blur, "*")))
        image_names_gt   = sorted(glob.glob(os.path.join(self.dir_image_gt, "*")))
        image_names_kernel = sorted(glob.glob(os.path.join(self.dir_image_kernel, "*")))
        return image_names_gt, image_names_blur, image_names_kernel

    def _load(self):

        data_blur = np.load(self.images_blur[0])
        data_sharp = np.load(self.images_sharp[0])
        data_kernel = np.load(self.images_kernel[0])
        return list(data_sharp), list(data_blur), list(data_kernel)

    def __getitem__(self, idx):

        blurs = self.data_blur[idx].transpose(2,0,1)
        sharp = self.data_sharp[idx].transpose(2,0,1)
        kernels = self.data_kernel[idx].transpose(2,0,1)
        return torch.from_numpy(blurs), torch.from_numpy(sharp), torch.from_numpy(kernels)

    def __len__(self):
        return self.num_image
