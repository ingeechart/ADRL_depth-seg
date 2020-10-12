# coding=<utf-8>
import os
from yacs.config import CfgNode as CN

_C = CN()

''' Define Experiments '''
_C.EXP = CN()
_C.EXP.NAME = 'test_model'
_C.EXP.OUTPUT_DIR = 'results/'
_C.EXP.LOG_DIR = 'results/looog.log'
_C.EXP.PRINT_FREQ = 10

''' Define system environment for Training '''
_C.SYS = CN()
_C.SYS.GPUS = (0,1,2,3)
_C.SYS.WORKERS = 8
_C.SYS.LOCAL_RANK = ''
_C.SYS.PIN_MEMORY = True

_C.SYS.CUDNN_DETERMINISTIC = True
_C.SYS.CUDNN_BENCHMARK = False
_C.SYS.CUDNN_ENABLED= True


''' Define Detail hyperparameters for training '''
_C.TRAIN = CN()
_C.TRAIN.SEED = 8967
_C.TRAIN.RESUME = ''
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 800
_C.TRAIN.BATCH_SIZE = 16

_C.TRAIN.OPT = CN()
_C.TRAIN.OPT.NAME = 'SGD'
_C.TRAIN.OPT.LR = 0.1
_C.TRAIN.OPT.WD = 1e-4
_C.TRAIN.OPT.MOMENTUM = 0.9

_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.CROP_SIZE = [320,480]
_C.TRAIN.DATA.SHUFFLE = False
_C.TRAIN.DATA.DROP_LAST = True
_C.TRAIN.DATA.AUG = ''



''' common params for NETWORK '''
_C.MODEL = CN()
_C.MODEL.NAME = 'hardnet'
_C.MODEL.BB_PRETRAINED = ''
_C.MODEL.FREEZE_BN = ''
_C.MODEL.NONBACKBONE_KEYWORDS = []
_C.MODEL.NONBACKBONE_MULT = 10

_C.MODEL.LOSS =CN()
_C.MODEL.LOSS.NAME = 'bootstrapped_cross_entropy'
_C.MODEL.LOSS.MIN_K = 4096
_C.MODEL.LOSS.THRESHOLD = 0.3
_C.MODEL.LOSS.SIZE_AVG = True

''' DATASET related params '''
_C.DATASET = CN()
_C.DATASET.ROOT = 'ADRL'
_C.DATASET.NAME = 'ADRL'
_C.DATASET.NUM_CLASSES = 4
_C.DATASET.TRAIN_SPLIT = 'train'
_C.DATASET.VAL_SPLIT = 'val'
_C.DATASET.TEST_SPLIT = 'test'



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
      print(sys.argv)
      print(_C, file=f)