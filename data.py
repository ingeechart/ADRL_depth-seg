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
class Cityscapes(data.Dataset):
    def __init__(self, data_root, split='train',transforms=False, ignore_label=255):
        '''
        path:
            Image: /cityscapes/leftImg8bits/train/
                    bochum/bochum_000000_000600_leftImg8bit.png

            ground truth: /cityscapes/gtFine_trainvaltest/gtFine/train/
                            bochum/bochum_000000_000313_gtFine_color.png

            root: ~/workspace/dataset/cityscapes
        '''
        self.root = data_root
        self.split = split

        self.image_base = os.path.join(self.root, 'leftImg8bit',self.split)
        self.files = glob.glob(self.image_base+'/*/*.png')
        assert len(self.files) is not 0 , ' cannot find datas!'
        
        self.transforms = transforms

        self.nClasses = 19

        self.colors = [  # [  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]
        label_colours = dict(zip(range(19), self.colors))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_path = self.files[index]
        img = Image.open(img_path)
        img = np.asarray(img)

        gt_path = img_path.replace('leftImg8bit','gtFine_trainvaltest/gtFine',1)
        # gt_path = gt_path.replace('leftImg8bit','gtFine_labelTrainIds',1)
        gt_path = gt_path.replace('leftImg8bit','gtFine_labelIds',1)

        label = Image.open(gt_path)
        label = np.asarray(label)


        if self.transforms:
            img, label= self.transforms(img,label)
        label = self.convert_label(label)

        return img, label


    def id2Trainid(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
    # the three channels

    def rgb2id(self, img)
        label_seg = np.zeros((img.shape[:2]), dtype=np.int)
        
        for i, id in enumerate(self.colors):
            label_seg[(img==id).all(axis=2)] = i+1
    '''
    def id2rgb(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def trainId2id(self, temp):
        ids = np.zeros((temp.shape[0], temp.shape[1]),dtype=np.uint8)
        for l in range(0, self.n_classes):
            ids[temp == l] = self.valid_classes[l]
        return ids
    '''
    def get_file_path(self):
        return self.files


def build_train_loader(cfg):

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # TODO
    train_transforms = transform.Compose([
        transform.RandScale([0.5,2]),
        transform.RandomHorizontalFlip(),
        transform.Crop(cfg.TRAIN.DATA.CROP_SIZE, crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])

    train_data = Cityscapes(data_root=cfg.DATASET.ROOT, split=cfg.DATASET.TRAIN_SPLIT, transforms=train_transforms)

    if dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_data)
    else:
        sampler =  None


    data_loader = torch.utils.data.DataLoader(train_data,
                            num_workers=cfg.SYS.WORKERS,
                            batch_size=cfg.DATASET.TRAIN.BATCH_SIZE//len(cfg.SYS.GPUS),
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
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ])
    val_data = Cityscapes(data_root=cfg.DATASET.ROOT, split=cfg.DATASET.VAL_SPLIT, transforms=val_transforms)
    if dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(val_data)
    else:
        sampler =  None

    data_loader = torch.utils.data.DataLoader(val_data,
                                            num_workers=4,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            sampler=sampler)

    return data_loader

def build_test_loader(H,W):

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transforms = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ])
    val_data = Cityscapes(data_root='cityscapes/', split='val', transforms=val_transforms)
      
    img_list = val_data.get_file_path()
    
    data_loader = torch.utils.data.DataLoader(val_data,
                                            num_workers=4,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            sampler=None)

    return data_loader, img_list



if __name__=='__main__':
    import torchvision
    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    from data import transform

    root='data/cityscapes/'
    # print(root)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([0.5,2]),
        transform.RandomHorizontalFlip(),
        transform.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])

    dataset = Cityscapes(root,transforms=train_transform)

    img, label =dataset.__getitem__(0)
    print(img.dtype)
    print(label.shape)
    fig_in = plt.figure()
    ax = fig_in.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig_in.add_subplot(1,2,2)
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