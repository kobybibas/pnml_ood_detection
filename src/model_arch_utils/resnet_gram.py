'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_arch_utils.gram_model_utils import G_p


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.collecting = False
        self.gram_feats = []

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def forward(self, x):
        t = self.conv1(x)
        out = F.relu(self.bn1(t))
        self.record(t)
        self.record(out)
        t = self.conv2(out)
        out = self.bn2(self.conv2(out))
        self.record(t)
        self.record(out)
        t = self.shortcut(x)
        out += t
        self.record(t)
        out = F.relu(out)
        self.record(out)
        return out

    def record(self, t):
        # For Gram ood detection
        if self.collecting:
            feature = [G_p(t, p=p) for p in range(1, 11)]
            feature = np.array(feature).transpose(1, 0, 2)  # shape=[samples, powers,feature]
            self.gram_feats.append(feature)

    def reset(self):
        self.gram_feats = []


class ResNetGram(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetGram, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # Added this method for pNML ood detection
        self.features_out = out.clone()

        y = self.linear(out)
        return y

    def load(self, path="../models/resnet_cifar10.pth"):
        tm = torch.load(path, map_location="cpu")
        self.load_state_dict(tm)

    def get_features(self):
        """
        Added this method for pNML ood detection
        :return:
        """
        return self.features_out

    def set_collecting(self, is_collecting: bool):
        for layer in itertools.chain(self.layer1, self.layer2, self.layer3, self.layer4):
            layer.collecting = is_collecting

    def reset(self):
        for layer in itertools.chain(self.layer1, self.layer2, self.layer3, self.layer4):
            layer.reset()

    def collect_gram_features(self):
        gram_feats_all = []
        for layer in itertools.chain(self.layer1, self.layer2, self.layer3, self.layer4):
            gram_feats = layer.gram_feats
            gram_feats_all += gram_feats
        self.reset()
        return gram_feats_all


def ResNet34Gram(num_c):
    return ResNetGram(BasicBlock, [3, 4, 6, 3], num_classes=num_c)


if __name__ == '__main__':
    model = ResNet34Gram(10)
    model.set_collecting(True)
