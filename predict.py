# Requirements:
# apt update && apt install -y ffmpeg
# pip install -U protobuf 
# pip install --upgrade google-cloud-speech
# Copy credentials *.json!

import os
import io
import sys
import datetime
import subprocess

import numpy as np
import cv2
import imageio

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.oauth2 import service_account

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
    return np.array(frames).astype(np.uint8)[12:-12][::len(frames) // n_frames]

def draw_transcroption(out, transcription):
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


credentials = service_account.Credentials.from_service_account_file(
    "/data/data/PicsArtHack-9e8fe9a284be.json")
client = speech.SpeechClient(credentials=credentials)
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='en-US')

trainer = Trainer(path="./data/resnet50_05BCE_no_CLR_50e_ADAM_no_weight")
trainer.load_state(mode="metric")

while True:
    file_path = sys.stdin.readline()[:-1]
    if not os.path.isfile(file_path):
        print(file_path)
        print("file not found")
        sys.exit(-1)

    if file_path.split(".")[-1] != "mp4":
        imgs = cv2.imread(file_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        imgs = np.array(imgs, dtype=np.uint8)
        out = trainer.predict_crop(imgs)
        cv2.imwrite('./data/segmented.png', out[0])

    else:
        imgs = split_video(filename, n_frames=20)
        outs = trainer.predict_crop(imgs)

        command = "ffmpeg -i ./data/hello_vid.mp4 -ab 160k -ac 1 -ar 16000 -vn ./data/audio.wav"
        subprocess.call(command, shell=True)

        file_name = "/data/data/picsart1/data/audio.wav"
        with io.open(file_name, 'rb') as audio_file:
            content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

        response = client.recognize(config, audio)
        transcription = [result.alternatives[0].transcript for result in response.results]
        if len(transcription) > 0:
            outs = draw_transcroption(outs, transcription)

        for i, out in enumerate(outs):
            cv2.imwrite("/data/segmented_%i.png" % i)

    print("done")