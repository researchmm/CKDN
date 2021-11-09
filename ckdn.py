import torch
import torch.nn as nn
import random

try:
        from torch.hub import load_state_dict_from_url
except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.k=3
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.head = 8
        self.qse_1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.qse_2 = self._make_layer(block, 64, layers[0])
        self.csp = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.inplanes = 64
        self.dte_1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.dte_2 = self._make_layer(block, 64, layers[0])
        self.aux_csp = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ = nn.Sequential(
                nn.Linear((512) * 1 * 1, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 1),)
        self.fc1_ = nn.Sequential(
                nn.Linear((512) * 1 * 1, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 1),)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, extract=False):

        rest1, dist1, rest2, ref1 = torch.chunk(x,4,dim=1)

        rest1 = self.qse_2(self.maxpool(self.qse_1(rest1)))
        rest2 = self.qse_2(self.maxpool(self.qse_1(rest2)))
        dist1 = self.dte_2(self.maxpool(self.dte_1(dist1)))
        ref1  = self.dte_2(self.maxpool(self.dte_1(ref1)))

        x = torch.cat((rest1-dist1,rest1-ref1),dim=0)

        x = self.csp(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dr,fr = torch.chunk(x,2,dim=0)
        x = self.aux_csp(rest1-rest2)
        x = self.avgpool(x)
        nr = torch.flatten(x, 1)
        diff = torch.flatten(((dist1-ref1)**2),1).mean(1,keepdim=True)

        dr = torch.sigmoid(self.fc_(dr))
        fr = torch.sigmoid(self.fc_(fr))
        nr = torch.sigmoid(self.fc1_(nr))

        out = torch.cat((dr,fr,nr,diff),dim=1)

        return out


def model(**kwargs):
    return  _resnet('resnet50', Bottleneck, [3, 4, 6, 3], True, True,**kwargs)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        keys = state_dict.keys()
        for key in list(keys):
            if 'conv1' in key:
                state_dict[key.replace('conv1','qse_1')] = state_dict[key]
                state_dict[key.replace('conv1','dte_1')] = state_dict[key]
            if 'layer1' in key:
                state_dict[key.replace('layer1','qse_2')] = state_dict[key]
                state_dict[key.replace('layer1','dte_2')] = state_dict[key]
            if 'layer2' in key:
                state_dict[key.replace('layer2','csp')] = state_dict[key]
                state_dict[key.replace('layer2','aux_csp')] = state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model

