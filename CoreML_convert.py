import re

import onnx
import torch
from onnx import onnx_pb
from onnx_coreml import convert

from model import *

def init_unet(state_dict):
    model = UnetMobilenetV2(pretrained=False, num_classes=1, num_filters=32, Dropout=.2)
    model.load_state_dict(state_dict["state_dict"])
    return model

IMG_SIZE = (320, 256)
TMP_ONNX = './data/tmp/tmp.onnx'
WEIGHT_PATH = "./data/mobilenetV2_model_interpolate/mobilenetV2_model_interpolate_checkpoint_metric.pth"
ML_MODEL = re.sub('\.pth$', '.mlmodel', WEIGHT_PATH)

#convert and save ONNX
model = init_unet(torch.load(WEIGHT_PATH, map_location=lambda storage, loc: storage))
torch.onnx.export(model,
                  torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]),
                  TMP_ONNX)

# Convert ONNX to CoreML model
model_file = open(TMP_ONNX, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
# 595 is the identifier of output.
coreml_model = convert(model_proto,
                       image_input_names=['0'],
                       image_output_names=['595'])
coreml_model.save(ML_MODEL)