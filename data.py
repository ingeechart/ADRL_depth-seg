# coding=<utf-8>
import os
import glob
import numpy as np

import torch
from torch.utils import data
import torch.distributed as dist

from utils import transform
from PIL import Image

'''
data loader for cityscapes dataset with panoptic annotation

'''


class ADRLscapes(data.Dataset):
    def __init__(self, data_root, split='train', transforms=False, ignore_label=255):
        '''
        path
            Image: /cityscapes/leftImg8bits/train/
                    bochum/bochum_000000_000600_leftImg8bit.png

            ground truth: /cityscapes/gtFine_trainvaltest/gtFine/train/
                            bochum/bochum_000000_000313_gtFine_color.png

            root: ~/workspace/dataset/cityscapes
        '''
        self.root = data_root
        self.split = split
        self.image_base = os.path.join(self.root, 'dataset1', self.split, 'input')
        #self.image_base = os.path.join(self.root, self.split, 'input')
        self.seg_file = glob.glob(self.image_base + '/*/*/*/*.png')
        print(len(self.seg_file))
        assert len(self.seg_file) is not 0, 'cannot find datas!'
        #
        self.transforms = transforms

    def __len__(self):  #
        return len(self.seg_file)

    def __getitem__(self, index):
        img_path = self.seg_file[index]
        img = Image.open(img_path)
        img = np.asarray(img)

        gt_path = img_path.replace('input', 'target', 1)
        gt_path = gt_path.replace('rgb', 'label', 1)
        label = Image.open(gt_path)
        label = np.asarray(label)

        if self.transforms:
            img, label = self.transforms(img, label)

        return img, label

    def get_file_path(self):
        return self.seg_file


# ---------------------------------------------------------------init-------------------------


def build_train_loader(cfg):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # TODO

    train_transforms = transform.Compose([
        transform.RandScale([0.5, 2]),  #
        transform.RandomHorizontalFlip(),  #
        transform.Crop(cfg.TRAIN.DATA.CROP_SIZE, crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    train_data = ADRLscapes(data_root=cfg.DATASET.ROOT, split=cfg.DATASET.TRAIN_SPLIT, transforms=train_transforms)

    if dist.is_initialized():  # ?
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_data)

    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(train_data,
                                              num_workers=cfg.SYS.WORKERS,
                                              batch_size=cfg.TRAIN.BATCH_SIZE // len(cfg.SYS.GPUS),
                                              shuffle=False,
                                              pin_memory=cfg.SYS.PIN_MEMORY,
                                              drop_last=True,
                                              sampler=sampler)

    return data_loader


def build_val_loader(cfg):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transforms = transform.Compose([
        #transform.GenerateTarget(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std),
    ])
    val_data = ADRLscapes(data_root=cfg.DATASET.ROOT, split=cfg.DATASET.VAL_SPLIT, transforms=val_transforms)
    if torch.distributed.is_available():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(val_data)
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(val_data,
                                              num_workers=4,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              sampler=sampler)

    return data_loader


def build_test_loader(H, W):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transforms = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])
    val_data = ADRLscapes(data_root=cfg.DATASET.ROOT, split='val', transforms=val_transforms)

    img_list = val_data.get_file_path()

    data_loader = torch.utils.data.DataLoader(val_data,
                                              num_workers=4,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              sampler=None)

    return data_loader, img_list


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    from data import transform

    #root = 'ADRL'
    root = 'ws'
    # print(root)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([0.5, 2]),
        transform.RandomHorizontalFlip(),
        transform.Crop([320, 480], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    val_transforms = transform.Compose([
        transform.ToTensor(),
      #  transform.GenerateTarget(),
        transform.Normalize(mean=mean, std=std),
    ])

    dataset = ADRLscapes(root, transforms=val_transforms)

    img, label = dataset.__getitem__(0)
    print("imgshape:",img.shape)
    print("labelshape:",label.shape)
    print(img.dtype)
    print(label.shape)
    fig_in = plt.figure()
    ax = fig_in.add_subplot(1, 2, 1)
    ax.imshow(img[0])
    ax = fig_in.add_subplot(1, 2, 2)
    ax.imshow(label)
    plt.show()
    # img_path = 'dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_044227_leftImg8bit.png'
    # gt_path = img_path.replace('leftImg8bit','gtFine_trainvaltest/gtFine',1)
    # inst_path = gt_path.replace('leftImg8bit','gtFine_instanceIds')
    # instance = Image.open(inst_path)

    # fig_in = plt.figure()
    # ax = fig_in.add_subplot(1,2,1)
    # ax.imshow(img)
    # ax = fig_in.add_subplot(1,2,2)
    # ax.imshow(instance)

    # fig = plt.figure()
    # rows=3
    # cols=3
    # for i, mask in enumerate(cmap):
    #     ax = fig.add_subplot(rows,cols, i+1)
    #     ax.imshow(mask)
    # # plt.show()

    # fig2 = plt.figure()
    # for i, mask in enumerate(ox):
    #     ax = fig2.add_subplot(rows,cols, i+1)
    #     ax.imshow(mask)
    # # plt.show()

    # fig3 = plt.figure()
    # for i, mask in enumerate(oy):
    #     ax = fig3.add_subplot(rows,cols, i+1)
    #     ax.imshow(mask)
    # plt.show()