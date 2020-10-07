import os
from yacs.config import CfgNode as CN

_C = CN()

''' Define Experiments '''
_C.EXP = CN()
_C.EXP.NAME = 'segmentation'
_C.EXP.OUTPUT_DIR = 'results/'
_C.EXP.LOG_DIR = 'results/looog.log'


''' Define system environment for Training '''
_C.SYS = CN()
_C.SYS.GPUS = (0,1,2,3)
_C.SYS.WORKERS = 8
_C.SYS.LOCAL_RANK = ''
_C.SYS.PIN_MEMORY = True

_C.SYS.CUDNN_DETERMINISTIC = True
_C.SYS.CUDNN_BENCHMARK = True
_C.SYS.CUDNN_ENABLED= True


''' Define Detail hyperparameters for training '''
_C.TRAIN = CN()
_C.TRAIN.RESUME = ''
_C.TRAIN.INIT_LR = 0.1
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 5e-4
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 400
_C.TRAIN.BATCH_SIZE = 16


''' common params for NETWORK '''
_C.MODEL = CN()
_C.MODEL.NAME = 'model'
_C.MODEL.BB_PRETRAINED = ''
_C.MODEL.LOSS = ''
_C.MODEL.FREEZE_BN = ''
_C.MODEL.NONBACKBONE_KEYWORDS = []
_C.MODEL.NONBACKBONE_MULT = 10
 

''' DATASET related params '''
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = ''
_C.DATASET.TEST_SET = ''


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