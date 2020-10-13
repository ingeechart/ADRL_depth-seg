'''
KAU-RML ingee hong
'''
# coding=<utf-8>
import argparse
import numpy as np
import os
import time
import random
import config

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim as optim
import matplotlib.pyplot as plt
#from cv_idmap import id2rgb
from models import get_model
from data import build_test_loader, build_val_loader
from utils.utils import update_config, get_logger, adjust_learning_rate, AverageMeter, intersectionAndUnionGPU
import cv2
if torch.__version__ <= '1.1.0':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter



def parse_args():
    ''' This is needed for torch.distributed.launch '''
    parser = argparse.ArgumentParser(description='Train instance classifier')
    # This is passed via launch.py
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--config', default=None, type=str, help='config file')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = config.get_cfg_defaults()
    cfg = update_config(cfg, args)

    return args, cfg

def main():

    args, cfg = parse_args()

    # *  define paths ( output, logger) * #
    if not os.path.exists(cfg.EXP.OUTPUT_DIR):
        os.makedirs(cfg.EXP.OUTPUT_DIR)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    logger_path = cfg.EXP.OUTPUT_DIR+cfg.EXP.NAME+'_'+timestamp +'.log'

    global logger, vWriter
    logger = get_logger(logger_path)
    vWriter = SummaryWriter(cfg.EXP.OUTPUT_DIR+'tb_data/val')

    # # * controll random seed * #
    torch.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)
    random.seed(cfg.TRAIN.SEED)


    # TODO: findout what cudnn options are
    cudnn.benchmark = cfg.SYS.CUDNN_BENCHMARK
    cudnn.deterministic = cfg.SYS.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYS.CUDNN_ENABLED


    msg = '[{time}]' 'starts evaluations setting '\
            '{exp_name}'.format(time = time.ctime(), exp_name = cfg.EXP.NAME)
    logger.info(msg)


    # * GPU env setup. * #
    #TODO : dist.world_size find must
    model = get_model(cfg)


    logger.info("=> creating model ...")

    # TODO: get model arch
    model_dict = model.state_dict()
    trained_path='results/best.pth.tar'
    best_dict = torch.load(trained_path, map_location='cpu')
    logger.info(set(model_dict)==set(best_dict['state_dict']))
    model_dict.update(best_dict['state_dict'])
    model.load_state_dict(model_dict, strict=True)



    # * define OPTIMIZER * #
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': cfg.TRAIN.OPT.LR}]
    optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.OPT.LR, momentum=cfg.TRAIN.OPT.MOMENTUM, weight_decay=cfg.TRAIN.OPT.WD)

    # TODO
    # * build DATALODER * #
    #train_loader = build_train_loader(cfg)
    test_loader = build_val_loader(cfg)
    print(len(test_loader))

    best_mIoU = 0
    best_epoch = 0
    logger.info('starts evaluating')
    #test(model, test_loader, cfg)
    loss_val, mIoU_val, mAcc_val, allAcc_val = validation(model, test_loader, cfg)

def test(model, test_loader, cfg):
    model.eval()
    with torch.no_grad():
        for i_iter, (imgs, target) in enumerate(test_loader):
            loss, prediction =  model(imgs, target)





def validation(model, val_loader, cfg):

    batch_time = AverageMeter('vBatch_Time', ':6.3f')
    data_time = AverageMeter('vData_Time', ':6.3f')
    loss_meter = AverageMeter('vLoss', ':.4e')

    intersection_meter = AverageMeter('vIoU')
    union_meter = AverageMeter('vUnion')
    target_meter = AverageMeter('vTarget')

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i_iter, (imgs, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            loss, prediction = model(imgs, target)

            n = imgs.size(0)
            if dist.is_available():
                loss = loss * n
                dist.all_reduce(loss)
                n = n*dist.get_world_size()
                loss = loss / n
            else:
                loss = torch.mean(loss)
            print(np.unique(prediction[0]))
            fig_in = plt.figure()
            ax = fig_in.add_subplot(1, 2, 1)
            ax.imshow(prediction[0])
            ax = fig_in.add_subplot(1, 2, 2)
            ax.imshow(target[0])
            plt.savefig('evals/predict/predict_target_'+str(i_iter)+'.png', dpi = 200)
            plt.close(fig_in)
            if i_iter > 1:
                exit()
            # intersection, union, target = intersectionAndUnionGPU(prediction, target, 4)
            # if dist.is_initialized():
            #     torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            # loss_meter.update(loss.item(), n)

            # batch_time.update(time.time() - end)
            # end = time.time()

            # if ((i_iter+1) % cfg.EXP.PRINT_FREQ) == 0 and dist.get_rank() == 0:
            #     logger.info('Test: [{}/{}] '
            #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
            #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
            #                 'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
            #                 'Accuracy {accuracy:.4f}.'.format(i_iter + 1, len(val_loader),
            #                                                 data_time=data_time,
            #                                                 batch_time=batch_time,
            #                                                 loss_meter=loss_meter,
            #                                                 accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if dist.get_rank()==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(4):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__=='__main__':

    main()
