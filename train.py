import os
import shutil
import pickle
import argparse

from tqdm import tqdm
import numpy as np
import cv2
from numpy.random import RandomState
#import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, remove_small_holes

import torch
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable

from model import *
from utils import *

train_path = "./data/train/"
path_images = list(map(
    lambda x: x.split('.')[0],
    filter(lambda x: x.endswith('.jpg'), os.listdir('./data/train/'))))
prng = RandomState(42)

path_images *= 3
prng.shuffle(path_images)
train_split = int(len(path_images)*.8)
train_images, val_images = path_images[:train_split], path_images[train_split:]

dataset = DatasetProcessor(
    train_path, train_images, as_torch_tensor=True, augmentations=True, mask_weight=True)
dataset_val = DatasetProcessor(
    train_path, val_images, as_torch_tensor=True, augmentations=True, mask_weight=True)

params = {
    #LR_finder
    "batch_size":18,
    "max_lr":.5,
    "loss_window":10, 
    "loss_growth_trsh":.5,
    "alpha":0.1,
    
    #fit
    "wd":0.,
    "freeze_encoder":True,
    "early_stop":20,
    "max_lr_decay":.8,
    "epoch":200,
    "learning_rate":1e-4,
    "bce_loss_weight":0.5,
    "reduce_lr_patience":0,
    "reduce_lr_factor":0,
    "CLR":0
}

model_type = "resnet101"
#model_type = "mobilenetV2"

model_params = {
    "directory":"./data/",
    "model":model_type,
    "model_name":"%s_model" % (model_type),
    "Dropout":.4,
    "device_idx":0,
    "pretrained":True,
    "num_classes":1,
    "num_filters":32,
    "reset":True,
    "ADAM":True
}

trainer = Trainer(**model_params)
if params["CLR"] != 0:
    trainer.LR_finder(dataset, **params)
    trainer.show_lr_finder_out(save_only=True)

trainer.fit(dataset, dataset_val, **params)
trainer.plot_trainer_history(mode="loss", save_only=True)
trainer.plot_trainer_history(mode="metric", save_only=True)