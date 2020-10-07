# ADRL_depth-seg
depth & segmenentation code for ADRL 


# Prepare Dataset
```
ln -s /link to dataset
```

# Training.
for Single-Node multi-process distributed training, use
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE YOUR_TRAINING_SCRIPT.py --arg1 --arg2
```
# Test.
```

```

# Visualize 
```
```

# Structure
```
base
├── README.md
├── train.py
│   ├── main
│   ├── train
│   ├── validation
│
├── data.py
│   ├── cityscapes_Dataset
│   ├── build_train_dataloader
│   ├── build_val_dataloader
│
├── models
│   ├── model.py
│   ├── build_model
│   ├── backbone 
│   ├── modules
│   ├── model
│   ├── loss_function
│
├── utils
│   ├── utils.py
│   │   ├── get_logger
│   │   ├── AverageMeter
│   │   ├── adjust_learning_rate
│   │   ├── intersectionAndUnion
│   │   ├── intersectionAndUnionGPU
│   │   
│   ├── transform.py
│   │   ├── usilts for transform data
│
├── config.py
│
├── result
│   ├── tb_data
│   │   ├── train
│   │   ├── val
│   │
│   ├── exp_name_logfile.log
│
├── cityscapes (ln -s /dataDisk/Datasets/cityscapes/)
```

# Todo
0. DataLoader
1. Training code
2. Test code
3. demo.py
4. data.py transform 부분 따로 빼고 싶은데......