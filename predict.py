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
import argparse

import numpy as np
import cv2
import imageio

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.oauth2 import service_account

from train import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--google_api_keys_path', type=str, default=False, required=False)
args = parser.parse_args()
globals().update(vars(args))

if google_api_keys_path:
    credentials = service_account.Credentials.from_service_account_file(google_api_keys_path)
    client = speech.SpeechClient(credentials=credentials)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')

trainer = Trainer(path=model_path)
trainer.load_state(mode="metric")

while True:
    file_path = sys.stdin.readline().strip()
    if not os.path.isfile(file_path):
        print(file_path)
        print("file not found")
        sys.exit(-1)

    if file_path.split(".")[-1] != "mp4":
        imgs = cv2.imread(file_path)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        imgs = np.array(imgs, dtype=np.uint8)
        out = trainer.predict_crop(imgs)
        cv2.imwrite('%s/segmented.png' % data_path, out[0])

    else:
        imgs = split_video(filename, n_frames=20)
        outs = trainer.predict_crop(imgs)

        command = "ffmpeg -i %s/video.mp4 -ab 160k -ac 1 -ar 16000 -vn %s/audio.wav" % (data_path, data_path)
        subprocess.call(command, shell=True)

        file_name = "%s/audio.wav" % data_path
        with io.open(file_name, 'rb') as audio_file:
            content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

        response = client.recognize(config, audio)
        transcription = [result.alternatives[0].transcript for result in response.results]
        if len(transcription) > 0:
            outs = draw_transcription(outs, transcription)

        for i, im in enumerate(outs):
            cv2.imwrite("%s/segmented_%i.png" % (data_path, i), im)

    print("done")