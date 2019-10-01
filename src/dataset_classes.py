import numpy as np
from torchvision.datasets import CIFAR10


class NoiseDataset(CIFAR10):
    """
    Create dataset with random noise images in the same structure of CIFAR10
    """

    def __init__(self, root, transform=None, target_transform=None):
        super(NoiseDataset, self).__init__(root, train=False,
                                           transform=transform, target_transform=target_transform,
                                           download=False)

        # Create random data and labels
        self.test_labels = [-1] * 1000
        self.test_data = np.random.randint(0, high=256, size=(1000, 32, 32, 3), dtype='uint8')
