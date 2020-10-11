'''
KAU-RML ingee hong
'''

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


from models import get_model
from data import build_train_loader, build_val_loader
from utils.utils import update_config, get_logger, adjust_learning_rate, AverageMeter, intersectionAndUnionGPU

if torch.__version__ <= '1.1.0':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter



def parse_args():
    ''' This is needed for torch.distributed.launch '''
    parser = argparse.ArgumentParser(description='Train instance classifier')
    # This is passed via launch.py
    parser.add_argument("--local_rank", type=int, default=0) #실행할 떄 multigpu --
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

    logger_path = 'results/'+cfg.EXP.NAME+'_'+timestamp +'.log'

    global logger, tWriter, vWriter
    logger = get_logger(logger_path)
    tWriter = SummaryWriter(cfg.EXP.OUTPUT_DIR+'tb_data/train')
    vWriter = SummaryWriter(cfg.EXP.OUTPUT_DIR+'tb_data/val')

    # # * controll random seed * #
    torch.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)
    random.seed(cfg.TRAIN.SEED)
    #랜덤값 고정 실험

    # TODO: findout what cudnn options are
    cudnn.benchmark = cfg.SYS.CUDNN_BENCHMARK #CUDNN_BENCHMARK 하나는 False 하나는 True
    cudnn.deterministic = cfg.SYS.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYS.CUDNN_ENABLED


    msg = '[{time}]' 'starts experiments setting '\
            '{exp_name}'.format(time = time.ctime(), exp_name = cfg.EXP.NAME)
    logger.info(msg)


    # * GPU env setup. * #
    distributed = args.local_rank >= 0 #local_rank 가 아니고 다른거 world_size로
    if distributed:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )


    # * define MODEL * #
    if dist.get_rank()==0:
        logger.info("=> creating model ...")

    # TODO: get model arch
    model = get_model(cfg) #backbone 모델 가져오는 부분
    pretrained_path='models/hardnet_petite_base.pth' #패스 지정
    weights = torch.load(pretrained_path) #weight를 불러오는 부분
    model.module.base.load_state_dict(weights)  #back bone 모델에 weight를 넣는 부분
    #encoding -> image에서 feature를 뽑아내는 부분 (backbone) , decoding -> feature에서 해석하는 부분 (header)
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # convert BN to syncBN 로 바꿔줘야 배치노말을 쓰는 것
        #그냥 BN을 쓰면 각자로 BN을 하는거고 sync BN을 써야 다 각자의 GPU에 있는 거 합쳐서 BN을 진행
        model = model.to(device)
        model = DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        ) # model을 각자의 Gpu로 보내는 부분
    else:
        model = nn.DataParallel(model, device_ids=cfg.gpus).cuda() #이건 syncBN을 못해. 그래서 distributedDataParallel을 씀

    if dist.get_rank()==0:
        logger.info(model) #model이 어떻게 생겼는지 기록이 남음


    # * define OPTIMIZER * #
    params_dict = dict(model.named_parameters()) #backbone과 header가 다르게 lr을 쓰고 싶으면 2개를 선언. model.backbone. 어쩌고 & model.header.파람
    params = [{'params': list(params_dict.values()), 'lr': cfg.TRAIN.OPT.LR}] #[ ~, ~] 꼴로 하면 opti를 2개 만들 수 있음
    optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.OPT.LR, momentum=cfg.TRAIN.OPT.MOMENTUM, weight_decay=cfg.TRAIN.OPT.WD)


    # * RESUME * # 다시 시작 할 수 있는 부분
    if cfg.TRAIN.RESUME:
        if os.path.isfile(cfg.TRAIN.RESUME):
            if dist.get_rank()==0:
                logger.info("=> loading checkpoint '{}'".format(cfg.TRAIN.RESUME))

            checkpoint = torch.load(cfg.TRAIN.RESUME, map_location=lambda storage, loc: storage.cuda())

            start_epoch = checkpoint['epoch']

            checkpoint_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            checkpoint_dict = {'module.'+k: v for k, v in checkpoint_dict.items()
                            if 'module.'+k in model_dict.keys()} #distributed 했을 떄 앞에 module. 이런 게 붙어서 추가시켜주는 중
            #이름이 어떻게 되어 있는지 확인을 하고 수정을 해야하는 부분

            for k, _ in checkpoint_dict.items():
                logger.info('=> loading {} pretrained model {}'.format(k, cfg.TRAIN.RESUME))
                #잘 들어가겠습니다 하고 알려줌
            logger.info(set(model_dict)==set(checkpoint_dict))
            assert set(model_dict)==set(checkpoint_dict) #다른게 있으면 assert로 error 표시

            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict, strict=True) #strict=True 시 model의 weight 이름, 불러온 weight 부분이 이름이 같아야함
            #strict = False시 이름이 같지 않으면 안들어감 대신 에러도 안내보냄. 같으면 들어가게 해줌.

            optimizer.load_state_dict(checkpoint['optimizer'])
            if dist.get_rank()==0:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAIN.RESUME, checkpoint['epoch']))
        else:
            if dist.get_rank()==0:
                logger.info("=> no checkpoint found at '{}'".format(cfg.TRAIN.RESUME))


    # TODO
    # * build DATALODER * #
    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg)


    best_mIoU = 0
    best_epoch = 0

    logger.info('starts training')
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.END_EPOCH):
        epoch_log = epoch+1
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(model, train_loader, optimizer, epoch, cfg)

        if args.local_rank <= 0:
            tWriter.add_scalar('loss', loss_train, epoch_log)
            tWriter.add_scalar('mIoU', mIoU_train, epoch_log)
            tWriter.add_scalar('mAcc', mAcc_train, epoch_log)
            tWriter.add_scalar('allAcc', allAcc_train, epoch_log)
            torch.save({ #checkpoint save 부분
                'epoch': epoch_log,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(cfg.EXP.OUTPUT_DIR,'checkpoint.pth.tar'))

        loss_val, mIoU_val, mAcc_val, allAcc_val = validation(model, val_loader, cfg)


        if args.local_rank <= 0:
            vWriter.add_scalar('loss', loss_val, epoch_log)
            vWriter.add_scalar('mIoU', mIoU_val, epoch_log)
            vWriter.add_scalar('mAcc', mAcc_val, epoch_log)
            vWriter.add_scalar('allAcc', allAcc_val, epoch_log)

            ## TODO 이 부분 한번 확인해야함. check point for best validation
            if best_mIoU < mIoU_val:
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(cfg.EXP.OUTPUT_DIR,'best.pth.tar'))
                best_mIoU = mIoU_val
                best_epoch = epoch+1

        if args.local_rank <= 0:
            msg = 'Loss_train: {:.10f}  Loss_val: {:.10f}'.format(loss_train, loss_val)
            logger.info(msg)
            msg = 'Best mIoU: {}  Best Epoch: {}'.format(best_mIoU, best_epoch)
            logger.info(msg)



    if args.local_rank <= 0:
        torch.save(model.module.state_dict(),
            os.path.join(cfg.EXP.OUTPUT_DIR, 'final_state.pth'))


def train(model, train_loader, optimizer, epoch, cfg):

    batch_time = AverageMeter('Batch_Time', ':6.3f')
    data_time = AverageMeter('Data_Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')

    intersection_meter = AverageMeter('IoU')
    union_meter = AverageMeter('Union')
    target_meter = AverageMeter('Target')

    model.train()
    max_iter = cfg.TRAIN.END_EPOCH * len(train_loader)
    end = time.time()

    for i_iter, (imgs, target) in enumerate(train_loader): #BCHW -> Tensor shape
        data_time.update(time.time() - end)
 
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss, predict = model(imgs, target) # returns loss and predictions at each GPU

        # model.zero_grad()
        optimizer.zero_grad()
        loss.backward() # distributed.datapaprallel automatically gather and syncronize losses.
        optimizer.step()


        # TODO. this tis for recordings. need to refine this.
        n = imgs.size(0) # n = batch size of each GPU
        if dist.is_initialized():
            loss = loss * n
            dist.all_reduce(loss)
            n = 2*dist.get_world_size()
            loss = loss / n
        loss_meter.update(loss.item(), n)


        # TODO 코드 확인. (evaluation metric), need to be changed with eval code of cityscape dataset
        #* 이부분이 정말 필요한가...? 연산도 이상하게 했다...
        intersection, union, target = intersectionAndUnionGPU(predict, target, 19, 255)
        if dist.is_initialized():
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)


        batch_time.update(time.time() - end)
        end = time.time()


        current_iter = epoch * len(train_loader) + i_iter + 1

        lr = adjust_learning_rate(optimizer,
                                  cfg.TRAIN.OPT.LR,
                                  max_iter,
                                  current_iter)

        # * compute remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        
        if (i_iter+1) % cfg.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg ='Epoch: [{}/{}][{}/{}] '\
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '\
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '\
                    'Remain {remain_time} '\
                    'Loss {loss_meter.val:.4f} '\
                    'Accuracy {accuracy:.4f} '\
                    'lr {lr}.'.format(epoch+1, cfg.TRAIN.END_EPOCH, i_iter + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        accuracy=accuracy,
                                                        lr = [x['lr'] for x in optimizer.param_groups])
            logger.info(msg)

        if dist.get_rank() == 0:
            tWriter.add_scalar('loss_batch', loss_meter.val, current_iter)
            tWriter.add_scalar('mIoU_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            tWriter.add_scalar('mAcc_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            tWriter.add_scalar('allAcc_batch', accuracy, current_iter)

        end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if dist.get_rank() == 0:
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, cfg.TRAIN.END_EPOCH, mIoU, mAcc, allAcc)) # max epoch: 200
    return loss_meter.avg, mIoU, mAcc, allAcc


def validation(model, val_loader, cfg):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i_iter, (imgs, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            imgs = imgs.cuda()
            target = target.cuda()

            loss, prediction = model(imgs, target)
    
            n = imgs.size(0)        
            if dist.is_initialized():
                loss = loss * n
                dist.all_reduce(loss)
                n = 2*dist.get_world_size()
                loss = loss / n
            else:
                loss = torch.mean(loss)

            intersection, union, target = intersectionAndUnionGPU(prediction, target, 19, 255)
            if dist.is_initialized():
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i_iter+1) % cfg.PRINT_FREQ) == 0 and dist.get_rank() == 0:
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(i_iter + 1, len(val_loader),
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if dist.get_rank()==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(19):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__=='__main__':
    main()
