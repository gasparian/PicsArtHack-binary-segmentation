from torch import nn, cat
import torchvision
import math

#########################################################################################
# TO DO:
# - add depthwise-seaprable convolution -> it will replace all Conv2d modules
# - batchnorms in decoder?
#
#########################################################################################

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activate=True, batchnorm=False):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.batchnorm = batchnorm
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            if self.batchnorm:
                #x = self.bn(x)
                pass
            x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, num_filters: int, batch_activate=False):
        super(ResidualBlock, self).__init__()
        self.batch_activate = batch_activate
        self.bn_rb_1 = nn.BatchNorm2d(in_channels)
        self.bn_rb_2 = nn.BatchNorm2d(num_filters)
        self.activation = nn.ReLU(inplace=True)
        self.conv_block = ConvRelu(in_channels, num_filters, activate=True, batchnorm=True)
        self.conv_block_na = ConvRelu(in_channels, num_filters, activate=False, batchnorm=False)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, inp):
#         x = self.bn_rb_1(inp)
#         x = self.activation(x)
        x = self.conv_block(inp)
        x = self.conv_block_na(x)
        x = x.add(inp)
        if self.batch_activate:
            #x = self.bn_rb_2(x)
            x = self.activation(x)
        return x

class DecoderBlockResnet(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True, res_blocks_dec=False):
        super(DecoderBlockResnet, self).__init__()
        self.in_channels = in_channels
        self.res_blocks_dec = res_blocks_dec

        if is_deconv:
            layers_list = [ConvRelu(in_channels, middle_channels, activate=True, batchnorm=False)]
            
            if self.res_blocks_dec:
                residual_blocks = [
                    ResidualBlock(middle_channels, middle_channels, batch_activate=False),
                    ResidualBlock(middle_channels, middle_channels, batch_activate=True)
                ]
                
                layers_list.extend(residual_blocks)
            
            layers_list.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if not self.res_blocks_dec:
                layers_list.append(nn.ReLU(inplace=True))
            
            self.block = nn.Sequential(*layers_list)

        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, 
                            mode='bilinear',
                            align_corners=True #torch 0.4 req.
                           ),
                ConvRelu(in_channels, middle_channels),
                #nn.BatchNorm2d(middle_channels),
                ConvRelu(middle_channels, out_channels),
                #nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.block(x)

class UnetResNet(nn.Module):

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv=True, 
                       Dropout=.2, model="resnet50"):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet50
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        
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
        self.res_blocks_dec = res_blocks_dec
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
                                         num_filters * 8, is_deconv, res_blocks_dec)
        self.dec5 = DecoderBlockResnet(self.filters_dict[model][1] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8, is_deconv, res_blocks_dec)    
        self.dec4 = DecoderBlockResnet(self.filters_dict[model][2] + num_filters * 8, 
                                       num_filters * 8 * 2, num_filters * 8, is_deconv, res_blocks_dec)
        self.dec3 = DecoderBlockResnet(self.filters_dict[model][3] + num_filters * 8, 
                                       num_filters * 4 * 2, num_filters * 2, is_deconv, res_blocks_dec)
        self.dec2 = DecoderBlockResnet(self.filters_dict[model][4] + num_filters * 2, 
                                       num_filters * 2 * 2, num_filters * 2 * 2, is_deconv, res_blocks_dec)
        
        self.dec1 = DecoderBlockResnet(num_filters * 2 * 2, num_filters * 2 * 2, 
                                       num_filters, is_deconv, res_blocks_dec)
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
        dec5 = self.dec5(cat([center, conv5], 1))
        dec4 = self.dec4(cat([dec5, conv4], 1))
        dec3 = self.dec3(cat([dec4, conv3], 1))
        dec2 = self.dec2(cat([dec3, conv2], 1))
        dec2 = self.dropout_2d(dec2)
            
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)

###########################################################################
# Mobile Net
###########################################################################

# Example
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
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