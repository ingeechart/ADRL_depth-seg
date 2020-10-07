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
├── model.py
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
├── config.py(Todo)
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
1. Model
2. Training code
3. Test code
