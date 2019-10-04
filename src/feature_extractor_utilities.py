import torch.nn.functional as F

from densenet import DenseNet3
from wideresnet import CIFARWRN


# Pretrained models:
# https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz
# https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz
# https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz
# https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet100.pth.tar.gz

class DensNetFeatureExtractor(DenseNet3):
    # Split the densnet: feature extractor and classifier

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        self.features = out.clone()
        return self.fc(out)

    def get_features(self):
        return self.features


class WideResNetFeatureExtractor(CIFARWRN):
    # Split the WideResNet: feature extractor and classifier

    def __init__(self, num_classes):
        blocks = 28
        layers = [(blocks - 4) // 6] * 3
        channels_per_layers = [16, 32, 64]
        init_block_channels = 16
        width_factor = 10
        channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]

        super().__init__(channels=channels,
                         init_block_channels=init_block_channels,
                         num_classes=num_classes)
        self.features_out = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self.features_out = x.clone()
        x = self.output(x)
        return x

    def get_features(self):
        return self.features_out
