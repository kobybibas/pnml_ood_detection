import torch.nn.functional as F

from densenet import DenseNet3
from loguru import logger
from torch import nn


# Pretrained models:
# https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz
# https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz
# https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz
# https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet100.pth.tar.gz

class DensNetSplit(DenseNet3):
    # Split the densnet: feature extractor and classifier

    def __init__(self, *args, **kwargs):
        super(DensNetSplit, self).__init__(*args, **kwargs)
        self.is_feature_extractor = True
        self.feature_extractor_layer = 0

        self.fc1 = nn.Linear(self.fc.in_features, self.fc.in_features)
        self.fc2 = nn.Linear(self.fc.in_features, self.fc.out_features)

        # convert later to model layer:
        self.conversion_dict = {
            '0': 0,  # conv1
            '1': 2,  # block1 + trans1
            '2': 4,  # block2 + trans2
            '3': 7,  # block3 + bn1 + relu
            '4': 8,  # fc
        }

    # def cuda(self, device=None):
    #     if self.is_feature_extractor is True:
    #         super().cuda()
    #     else:
    #         max_freeze_layer = self.conversion_dict[str(self.feature_extractor_layer)]
    #         for ct, child in enumerate(self.children()):
    #             if ct <= max_freeze_layer:
    #                 # logger.debug('Freeze Layer: idx={}, name={}'.format(ct, child))
    #                 # for param in child.parameters():
    #                 #     param.requires_grad = False
    #                 # continue
    #                 child.cpu()
    #                 continue
    #             # logger.debug('UnFreeze Layer: idx={}, name={}'.format(ct, child))
    #             child.cuda()
    #     return self

    def set_feature_extractor_layer(self, layer_num):
        assert layer_num >= 0
        assert layer_num <= 4
        self.feature_extractor_layer = layer_num

    def convert_to_feature_extractor(self):
        self.is_feature_extractor = True
        self.eval()

        max_freeze_layer = self.conversion_dict[str(self.feature_extractor_layer)]
        for ct, child in enumerate(self.children()):
            if ct <= max_freeze_layer:
                logger.debug('Freeze Layer: idx={}, name={}'.format(ct, child))
                for param in child.parameters():
                    param.requires_grad = False
                continue
            logger.debug('UnFreeze Layer: idx={}, name={}'.format(ct, child))

    def convert_to_classifier(self):
        self.is_feature_extractor = False

    def forward(self, x):
        out = x

        # Layer 0
        if self.is_feature_extractor is True or self.feature_extractor_layer < 0:
            # print('Layer 0')
            out = self.conv1(out)

        if self.is_feature_extractor is True and self.feature_extractor_layer == 0:
            return out

        # Layer 1
        if self.is_feature_extractor is True or self.feature_extractor_layer < 1:
            # print('Layer 1')
            out = self.trans1(self.block1(out))

        if self.is_feature_extractor is True and self.feature_extractor_layer == 1:
            return out

        # Layer 2
        if self.is_feature_extractor is True or self.feature_extractor_layer < 2:
            # print('Layer 2')
            out = self.trans2(self.block2(out))

        if self.is_feature_extractor is True and self.feature_extractor_layer == 2:
            return out

        # Layer 3
        if self.is_feature_extractor is True or self.feature_extractor_layer < 3:
            # print('Layer 3')
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.in_planes)

        if self.is_feature_extractor is True and self.feature_extractor_layer == 3:
            return out

        # # Layer 4
        # if self.is_feature_extractor is True or self.feature_extractor_layer < 4:
        #     # print('Layer 4')
        #     out = self.fc(out)

        # todo: I increased the number of parameters here
        # Layer 4
        if self.is_feature_extractor is True or self.feature_extractor_layer < 4:
            # print('Layer 4')
            out = self.fc1(out)
            out = self.fc2(out)

        return out

    def reset_fc(self):
        self.fc1 = nn.Linear(self.fc.in_features, self.fc.in_features)
        self.fc2 = nn.Linear(self.fc.in_features, self.fc.out_features)

    def forward_super(self, x):
        return super().forward(x)
