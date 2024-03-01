import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from .resnet_agw import resnet50


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return "input", "backward" will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class AdversarialLayer(nn.Module):
    def __init__(self, per_add_iters, iter_num=0, alpha=10.0, low_value=0.0, high_value=1.0, max_iter=10000.0):
        super(AdversarialLayer, self).__init__()
        self.per_add_iters = per_add_iters
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter
        self.grl = GradientReversalLayer.apply

    def forward(self, input, train_set=True):
        if train_set:
            self.iter_num += self.per_add_iters
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                    self.high_value - self.low_value) + self.low_value)

        return self.grl(self.coeff, input)


class DiscriminateNet(nn.Module):
    def __init__(self, input_dim, class_num=1, simple=False):
        super(DiscriminateNet, self).__init__()
        self.input_dim = input_dim
        self.class_num = class_num
        self.simple = simple
        if self.simple == True:
            self.ad_layer = nn.Linear(input_dim, class_num)
            self.ad_layer.apply(weights_init_classifier)
        else:
            self.ad_layer1 = nn.Linear(input_dim, input_dim // 2)
            self.ad_layer2 = nn.Linear(input_dim // 2, input_dim // 2)
            self.ad_layer3 = nn.Linear(input_dim // 2, class_num)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.5)
            self.dropout2 = nn.Dropout(0.5)
            self.bn2 = nn.BatchNorm1d(input_dim // 2)
            self.bn2.bias.requires_grad_(False)
            self.ad_layer1.apply(weights_init_kaiming)
            self.ad_layer2.apply(weights_init_kaiming)
            self.ad_layer3.apply(weights_init_classifier)
        self.bn = nn.BatchNorm1d(class_num)
        self.bn.bias.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.simple:
            x = self.ad_layer(x)
        else:
            x = self.ad_layer1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.ad_layer2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.ad_layer3(x)
        x = self.bn(x)
        x = self.sigmoid(x)  # binary classification

        return x


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x, mode='sh'):
        if mode == 'sh':
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)
            x1 = self.visible.layer1(x)
            x2 = self.visible.layer2(x1)
            return x2
        else:
            x3 = self.visible.layer3(x)
            x4 = self.visible.layer4(x3)

            return x4, x3


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()
        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t

    def forward(self, x, mode='sh'):
        if mode == 'sh':
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)
            x1 = self.thermal.layer1(x)
            x2 = self.thermal.layer2(x1)
            return x2
        else:
            x3 = self.thermal.layer3(x)
            x4 = self.thermal.layer4(x3)

            return x4, x3


class share_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(share_module, self).__init__()
        model_share = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        model_share.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share = model_share

    def forward(self, x):
        x = self.share.layer3(x)
        x = self.share.layer4(x)
        return x


class network(nn.Module):
    def __init__(self, num_classes=1000, arch='resnet50', gm_pool='on'):
        super(network, self).__init__()

        self.visible_module = visible_module(arch=arch)
        self.thermal_module = thermal_module(arch=arch)
        self.share_module = share_module(arch=arch)

        pool_dim = 2048
        self.num_features = pool_dim

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.adnet = AdversarialLayer(per_add_iters=1.0)
        self.disnet = DiscriminateNet(pool_dim, 1)
        self.gm_pool = gm_pool

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x_sh = self.share_module(x)

        # pooling
        if self.gm_pool == 'on':
            b, c, h, w = x_sh.shape
            x_sh = x_sh.view(b, c, -1)
            p = 3.0
            x_pool_sh = (torch.mean(x_sh ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool_sh = self.avgpool(x_sh)
            x_pool_sh = x_pool_sh.view(x_pool_sh.size(0), x_pool_sh.size(1))

        feat = self.bottleneck(x_pool_sh)

        if self.training:
            x_dis = self.disnet(self.adnet(feat))
            return feat, x_dis
        else:
            return self.l2norm(feat)


def res(pretrained=False, gm_pool='one', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = network(arch='resnet50', gm_pool='on')

    return model
