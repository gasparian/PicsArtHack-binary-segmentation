import os
import io
import sys
import time
import datetime
import subprocess
import argparse

import numpy as np
import cv2

from utils import *

# python3 predict.py -p ./test --model_path ./models/model --gpu -1

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--data_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--delay', type=int, default=7, required=False)
parser.add_argument('--denoise_borders', action='store_true')
parser.add_argument('--gpu', type=int, default=-1, required=False)
args = parser.parse_args()
globals().update(vars(args))

trainer = Trainer(path=model_path, gpu=gpu)
trainer.load_state(mode="metric")

files_list = os.listdir(data_path)

images, vids = [], []
if files_list:
    for fname in files_list:    
        if fname.split(".")[-1] != "mp4": images.append(fname)
        elif fname.split(".")[-1] == "mp4": vids.append(fname)

    if images:
        for fname in images:
            img = cv2.imread(data_path+"/"+fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.uint8)
            out = trainer.predict_mask(img, biggest_side=None, denoise_borders=denoise_borders)
            cv2.imwrite('%s/%s_seg.png' % (data_path, fname.split(".")[0]), out[0])
        print(" [INFO] Images processed! ")

    if vids:
        for fname in vids:
            imgs = split_video(data_path+"/"+fname, frame_rate=12)
            out = trainer.predict_mask(imgs, biggest_side=None, denoise_borders=denoise_borders)
            vpath = data_path+"/%s" % fname.split(".")[0]
            os.mkdir(vpath)
            save_images(out, path=vpath)
            os.system(f"convert -delay {delay} -loop 0 -dispose Background {vpath}/*.png {vpath}/{fname.split('.')[0]}.gif")
        print(" [INFO] Videos processed! ")

print(" [INFO] %s ms. " % round((time.time()-start)*1000, 0))