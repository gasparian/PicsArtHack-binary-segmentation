import sys
import os

import numpy as np
import cv2

from train import *

def split_video(filename, n_frames=20):
    vidcap = cv2.VideoCapture(filename)
    frames = []
    succ, frame = vidcap.read()
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    while succ:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame[:, ::-1, :], axes=[1,0,2])
        frames.append(frame)
        succ, frame = vidcap.read()
    return np.array(frames).astype(np.uint8)[12::len(frames) // n_frames]

trainer = Trainer(path="./data/resnet50_05BCE_no_CLR_50e_ADAM_no_weight")
trainer.load_state(mode="metric")

while True:
    file_path = sys.stdin.readline()[:-1]
    if not os.path.isfile(file_path):
        print(file_path)
        print("file not found")
        sys.exit(-1)

    if file_path.split(".")[-1] == "png":
        imgs = cv2.imread(file_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        imgs = np.array(imgs, dtype=np.uint8)
        out = trainer.predict_crop(imgs)
        cv2.imwrite('./data/segmented.png', out[0])

    else:
        imgs = split_video(filename)
        outs = trainer.predict_crop(imgs)
        for i, out in enumerate(outs):
            cv2.imwrite("/data/segmented_%i.png" % i)

    print("done")