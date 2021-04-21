#!/bin/bash

declare -a trainsets=("cifar10" "cifar100" "svhn")
declare -a models=("densenet" "resnet")

cd ../src || exit

for trainset in ${trainsets[@]}; do
  for model in ${models[@]}; do
    echo $model $trainset
    python optimize_odin.py model=$model trainset=$trainset
  done
done
