import re
import argparse

import onnx
import torch
from onnx import onnx_pb
from onnx_coreml import convert

from model import *

#https://github.com/akirasosa/mobile-semantic-segmentation/blob/master/coreml_converter.py

# python3 CoreML_convert.py --tmp_onnx ./models/tmp.onnx  --weights_path ./models/mobilenetV2_model/mobilenetV2_model_checkpoint_metric.pth

def init_unet(state_dict):
    model = UnetMobilenetV2(pretrained=False, num_classes=1, num_filters=32, Dropout=.2)
    model.load_state_dict(state_dict["state_dict"])
    return model

parser = argparse.ArgumentParser(description='crnn_ctc_loss')
parser.add_argument('--tmp_onnx', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--img_H', type=int, default= 320)
parser.add_argument('--img_W', type=int, default= 256)
args = parser.parse_args()
globals().update(vars(args))

coreml_path = re.sub('\.pth$', '.mlmodel', weights_path)

#convert and save ONNX
model = init_unet(torch.load(weights_path, map_location=lambda storage, loc: storage))
torch.onnx.export(model,
                  torch.randn(1, 3, img_H, img_W),
                  tmp_onnx)

# Convert ONNX to CoreML model
model_file = open(tmp_onnx, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
# 595 is the identifier of output.
coreml_model = convert(model_proto,
                       image_input_names=['0'],
                       image_output_names=['595'])
coreml_model.save(coreml_path)