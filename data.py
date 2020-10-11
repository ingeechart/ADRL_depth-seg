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


class ADRLscapes(data.Dataset):  # 데이터셋
    def __init__(self, data_root, split='train', transforms=False, ignore_label=255):
        '''
        path: ㅅㅈ
            Image: /cityscapes/leftImg8bits/train/
                    bochum/bochum_000000_000600_leftImg8bit.png

            ground truth: /cityscapes/gtFine_trainvaltest/gtFine/train/
                            bochum/bochum_000000_000313_gtFine_color.png

            root: ~/workspace/dataset/cityscapes
        '''
        # 파라미터들
        # 같은 아이디 셋
        self.root = data_root  # 리눅스 ㅅㅈ
        self.split = split
        self.image_base = os.path.join(self.root, 'dataset1', self.split)
        print(self.image_base)# ㅍ패스
        self.seg_file = glob.glob(self.image_base + '/*/Final_2/*/left_20.png')  # 별에는 뭐가 들어와도 ㄱㅊ
        print(len(self.seg_file))
        # seg_file 내에는 모든(left, right) rgb img file이 들어감
        # self.depth_file  = glob.glob(self.image_base+'/*/*.pfm') #depth
        assert len(self.seg_file) is not 0, 'cannot find datas!'
        # 수정 ? IS / ISNOT ?
        # 몇개의 파일을 가지는지 저장
        # 무작위ㅜ로 가져오니까 아무것도 없으면 어서트 해야됨 안그럼 가만히 있어 암것도 없어도

        self.transforms = transforms

        self.nClasses = 4

    def __len__(self):  # 길이
        return len(self.seg_file)

    def __getitem__(self, index):  # 부르면 나옴 ? 무조건 ?

        img_path = self.seg_file[index]  # 파일리스트에서 1개씩 가져옴
        print("img_path: ", img_path)
        img = Image.open(img_path)  # 이미지 불로
        img = np.asarray(img)  # 넘파이

        gt_path = img_path.replace('train', 'val', 1)  # 폴더이르,ㅁ ㅅㅈ
        # gt_path = gt_path.replace('leftImg8bit','gtFine_labelTrainIds',1)
        gt_path = gt_path.replace('rgb','label',1)
        print("gt_path: ", gt_path)

        label = Image.open(gt_path)
        label = np.asarray(label)  # pfm 도 넘파이로 ?
        # 넘파이 -> 트랜스펌

        #if self.transforms:
        #    img, label = self.transforms(img, label)
        #label = self.convert_label(label)  # 컨버팅 ?
        # 색맵을 클래스맵

        return img, label

    def get_file_path(self):
        return self.seg_file


# ---------------------------------------------------------------init-------------------------
# 트레인에 슬
# 로더3개

def build_train_loader(cfg):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]  # 이미지넷트레인 인풋 노말라이즈 스탠다드 ... 값

    # TODO
    # transform - 아큐멘테이션
    train_transforms = transform.Compose([  # 합쳐줌
        transform.RandScale([0.5, 2]),  # 0.5와 2사이로 잘라줌
        transform.RandomHorizontalFlip(),  # 반전 좡
        transform.Crop(cfg.TRAIN.DATA.CROP_SIZE, crop_type='rand', padding=mean, ignore_label=255),
        # 크롬 , 제로ㅍ딩, 민은 가장자리의 평균?
        transform.ToTensor(),  # 텐서로 바꿔야지 다른객체였다가 ,, 텐서로 만들어야지 토치텐서타입으로 바꿔서 사용
        transform.Normalize(mean=mean, std=std)
    ])  # 컨피그 트레인쪽에있음 파라미터 , 트레인, 어규

    train_data = ADRLscapes(data_root=cfg.DATASET.ROOT, split=cfg.DATASET.TRAIN_SPLIT, transforms=train_transforms)
    # LEN말고 GETIMG로 리턴받음

    if dist.is_initialized():  # ?
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_data)
        # 샘플러 = 디스트리뷰티드 할때 필용
        # 디스트리 ㄴ 샘플러 - 콜하면 하나싞불러ㄷ주는데
        # 콜할때 데이터ㅔㅅㅅ을 몇번 콜할지 0 배시사이즈만큼ㄴ 콜해줌
        # 결과가 3개 나옴
        # 따로나오면 하나의 텐ㄴ서로
        # 겟아이템 =이미지, 라벨 -> 1개 의 샘플
        # 디스트리면 4갸면 4차원 텐서
        # 콜레이트 BN - 나중에
    else:
        sampler = None

        # 겟아이탬을 이터레이트 하면서 함 배치사이즈만큼씩 이터
        # 배치사이즈 4인 텐서 , , , , , ,
    data_loader = torch.utils.data.DataLoader(train_data,
                                              num_workers=cfg.SYS.WORKERS,  # 몇명의 CPU스레드를 종으로 부리냐,
                                              # 1개당 4~8 ,, 초과하면 에러남 , 스레드 얘 전용 할당 워카
                                              batch_size=cfg.DATASET.TRAIN.BATCH_SIZE // len(cfg.SYS.GPUS),
                                              # 디스트리뷰티드 ,, 지피유당 가질 배치사이즈 넣어줌
                                              # 총 / 지피유수
                                              shuffle=False,  # 디스트리부트때 셔플 안씀
                                              pin_memory=cfg.SYS.PIN_MEMORY,  # 트루면 빠름
                                              drop_last=True,  # 디스틀부트에 트루
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
    val_data = ADRLscapes(data_root=cfg.DATASET.ROOT, split=cfg.DATASET.VAL_SPLIT, transforms=val_transforms)
    if dist.is_initialized():
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
    val_data = ADRLscapes(data_root='cityscapes/', split='val', transforms=val_transforms)

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
        transform.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    dataset = ADRLscapes(root, transforms=train_transform)

    img, label = dataset.__getitem__(0)
    print(img.dtype)
    print(label.shape)
    fig_in = plt.figure()
    ax = fig_in.add_subplot(1, 2, 1)
    ax.imshow(img)
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