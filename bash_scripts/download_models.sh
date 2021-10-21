#!/usr/bin/env bash

mkdir ../models
cd ../models || exit

wget https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth
wget https://www.dropbox.com/s/7ur9qo81u30od36/densenet_cifar100.pth
wget https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth
wget https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth
wget https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth
wget https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth
wget https://www.dropbox.com/s/elfw7e3uofpydg5/wideresnet100.pth.tar.gz
tar -xzvf wideresnet100.pth.tar.gz
rm wideresnet100.pth.tar.gz

wget https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz
tar -xzvf wideresnet10.pth.tar.gz
rm wideresnet10.pth.tar.gz

wget https://github.com/nazim1021/OOD-detection-using-OECC/blob/master/zeroshot_ood_experiments/results/Zero_Shot/Zero_Shot_cifar100_densenet_OECC_tune_epoch_9.pth?raw=true
wget https://github.com/nazim1021/OOD-detection-using-OECC/blob/master/zeroshot_ood_experiments/results/Zero_Shot/Zero_Shot_cifar10_densenet_OECC_tune_epoch_14.pth?raw=true
wget https://github.com/nazim1021/OOD-detection-using-OECC/blob/master/zeroshot_ood_experiments/results/Zero_Shot/Zero_Shot_svhn_densenet_OECC_tune_epoch_4.pth?raw=true
