from __future__ import print_function, division
import os

import cv2
from tqdm import tqdm
import numpy as np

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    ShiftScaleRotate,
    RandomBrightness
)


import torch
from torchvision import transforms
from torch.utils import data

class DatasetProcessor(data.Dataset):
    
    def __init__(self, root_path, file_list, is_test=False, as_torch_tensor=True, augmentations=False, mask_weight=True):
        self.is_test = is_test
        self.mask_weight = mask_weight
        self.root_path = root_path
        self.file_list = file_list
        self.as_torch_tensor = as_torch_tensor
        self.augmentations = augmentations
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.been = []
        
    def clear_buff(self):
        self.been = []
    
    def __len__(self):
        return len(self.file_list)

    def transform(self, image, mask):
        aug = Compose([
            HorizontalFlip(p=0.9),
            RandomBrightness(p=.5,limit=0.3),
            RandomContrast(p=.5,limit=0.3),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, 
                             p=0.7,  border_mode=0, interpolation=4)
        ])
        
        augmented = aug(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def get_mask_weight(self, mask):
        mask_ = cv2.erode(mask, kernel=np.ones((8,8),np.uint8), iterations=1)
        mask_ = mask-mask_
        return mask_ + 1
    
    def __getitem__(self, index):
        
        file_id = index
        if type(index) != str:
            file_id = self.file_list[index]
        
        image_folder = self.root_path
        image_path = os.path.join(image_folder, file_id + ".jpg")
        
        mask_folder = self.root_path[:-1] + "_mask/"
        mask_path = os.path.join(mask_folder, file_id + ".png")
        
        if self.as_torch_tensor:
                    
            if not self.is_test:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                #resize to 320x256
                image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                mask = cv2.resize(mask, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image, mask = self.transform(image, mask)
                    
                mask = mask // 255
                mask = mask[:, :, np.newaxis]
                if self.mask_weight:
                    mask_w = self.get_mask_weight(np.squeeze(mask))
                else: 
                    mask_w = np.ones((mask.shape[:-1]))
                mask_w = mask_w[:, :, np.newaxis]
                    
                mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32'))
                mask_w = torch.from_numpy(np.transpose(mask_w, (2, 0, 1)).astype('float32'))
                image = self.norm(image)
                return image, mask, list([file_id]), mask_w

            else:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 320), interpolation=cv2.INTER_LANCZOS4)
                image = self.norm(image)
                return image, list([file_id])
            
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.uint8)
            if not self.is_test:
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image, mask = self.transform(image, mask)
                return image, mask, file_id
            
            else:
                if self.augmentations:
                    if file_id not in self.been:
                        self.been.append(file_id)
                    else:
                        image = self.transform(image)
                return image, file_id

def split_video(filename, n_frames=20):
    vidcap = cv2.VideoCapture(filename)
    frames = []
    succ, frame = vidcap.read()
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    while succ:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = np.transpose(frame[:, ::-1, :], axes=[1,0,2])
        frames.append(frame)
        succ, frame = vidcap.read()
    return np.array(frames).astype(np.uint8)[12:-12][::len(frames) // n_frames]

def draw_transcription(out, transcription):
    out_tmp = out.copy()
    offset = 5
    word_duration = len(out) // len(transcription)
    scales = np.linspace(.1, 4, num=15)
    word_id = 0
    for n in range(out_tmp.shape[0]):
        if n == word_duration:
            word_id = min(len(transcription)-1, word_id + 1)
        max_w_h = np.where(out_tmp[n, :, :, 3].sum(axis=1))[0][0] - offset*2
        max_w_w = out_tmp[n].shape[1] - offset*2
        for s in range(scales.shape[0]):
            w_w_est, w_h_est = cv2.getTextSize(transcription[word_id],cv2.FONT_HERSHEY_TRIPLEX,scales[s],3)[0]
            if w_w_est > max_w_w or w_h_est > max_w_h:
                idx = max(0, s-1)
                w_w_est, w_h_est = cv2.getTextSize(transcription[word_id],cv2.FONT_HERSHEY_TRIPLEX,scales[idx],3)[0]
                break

        font = cv2.FONT_HERSHEY_TRIPLEX
        out_tmp[n] = cv2.putText(out_tmp[n], transcription[word_id], ((max_w_w - w_w_est) // 2,max_w_h), 
                                 font, scales[idx], (0,0,0,255), 3, cv2.LINE_AA)
    return out_tmp