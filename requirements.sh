#!/bin/bash
apt update && apt install ffmpeg imagemagick
pip3 --no-cache-dir install --upgrade tqdm==4.28.1 \
                                      numpy==1.14.3 \
                                      scikit-image==0.13.1 \
                                      albumentations==0.1.7 \
                                      opencv-python==3.4.3.18 \
                                      torch==0.4.1 \
                                      torchvision==0.2.1 \
                                      onnx==1.3.0 \
                                      six==1.10.0 \
                                      onnx-coreml