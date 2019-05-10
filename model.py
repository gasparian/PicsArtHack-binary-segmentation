import math

import torch
from torch import nn, cat
import torchvision

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activate=True):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, num_filters: int, batch_activate=False):
        super(ResidualBlock, self).__init__()
        self.batch_activate = batch_activate
        self.activation = nn.ReLU(inplace=True)
        self.conv_block = ConvRelu(in_channels, num_filters, activate=True)
        self.conv_block_na = ConvRelu(in_channels, num_filters, activate=False)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, inp):
        x = self.conv_block(inp)
        x = self.conv_block_na(x)
        x = x.add(inp)
        if self.batch_activate:
            x = self.activation(x)
        return x

class DecoderBlockResnet(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockResnet, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels, activate=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UnetResNet(nn.Module):

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, Dropout=.2, model="resnet50"):
        
        super().__init__()
        if model == "resnet18":
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
        elif model == "resnet34":
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif model == "resnet50":
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        elif model == "resnet101":
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            
        if model in ["resnet18", "resnet34"]: model = "resnet18-34"
        else: model = "resnet50-101"
            
        self.filters_dict = {
            "resnet18-34": [512, 512, 256, 128, 64],
            "resnet50-101": [2048, 2048, 1024, 512, 256]
        }
        
        self.num_classes = num_classes
        self.Dropout = Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        
        self.center = DecoderBlockResnet(self.filters_dict[model][0], num_filters * 8 * 2, 
                                         num_filters * 8)
        self.dec5 = DecoderBlockResnet(self.filters_dict[model][1] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8)    
        self.dec4 = DecoderBlockResnet(self.filters_dict[model][2] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlockResnet(self.filters_dict[model][3] + num_filters * 8, 
                                       num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlockResnet(self.filters_dict[model][4] + num_filters * 2, 
                                       num_filters * 2 * 2, num_filters * 2 * 2)
        
        self.dec1 = DecoderBlockResnet(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.dropout_2d = nn.Dropout2d(p=self.Dropout)
        

    def forward(self, x, z=None):
        conv1 = self.conv1(x)
        conv2 = self.dropout_2d(self.conv2(conv1))
        conv3 = self.dropout_2d(self.conv3(conv2))
        conv4 = self.dropout_2d(self.conv4(conv3))
        conv5 = self.dropout_2d(self.conv5(conv4))

        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec2 = self.dropout_2d(dec2)
            
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)

###########################################################################
# Mobile Net
###########################################################################

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):

    """
    from MobileNetV2 import MobileNetV2

    net = MobileNetV2(n_class=1000)
    state_dict = torch.load('mobilenetv2.pth.tar') # add map_location='cpu' if no gpu
    net.load_state_dict(state_dict)
    """

    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class UnetMobilenetV2(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True,
                 Dropout=.2, path='./data/mobilenet_v2.pth.tar'):
        super(UnetMobilenetV2, self).__init__()
        
        self.encoder = MobileNetV2(n_class=1000)
        
        self.num_classes = num_classes

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)

        #doesn't needed; obly for compatibility
        self.dconv_final = nn.ConvTranspose2d(1, 1, 4, padding=1, stride=2)

        if pretrained:
            state_dict = torch.load(path)
            self.encoder.load_state_dict(state_dict)
        else: self._init_weights()

    def forward(self, x):
        for n in range(0, 2):
            x = self.encoder.features[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.encoder.features[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.encoder.features[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.encoder.features[n](x)
        x4 = x

        for n in range(14, 19):
            x = self.encoder.features[n](x)
        x5 = x
        
        up1 = torch.cat([
            x4,
            self.dconv1(x)
        ], dim=1)
        up1 = self.invres1(up1)

        up2 = torch.cat([
            x3,
            self.dconv2(up1)
        ], dim=1)
        up2 = self.invres2(up2)

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        up3 = self.invres3(up3)

        up4 = torch.cat([
            x1,
            self.dconv4(up3)
        ], dim=1)
        up4 = self.invres4(up4)
        x = self.conv_last(up4)
        x = self.conv_score(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()