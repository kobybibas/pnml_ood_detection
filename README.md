# pnml_ood


Install requirements: 
```bash
while read requirement; conda install --yes $requirement;or pip install $requirement; end < requirements.txt
```

Download datasets:
```bash
cd data
chmod 777 ./download_data.sh
./download_data.sh
```

Steps:
1. Train model.
2. Extract features and logits.
3. Compute In-Distribution/OutOfDistribution score.
4. Calculate performance.

## 1. Train Model
There are pretrained model in here:
TODO
Simply download and extract to model directory.
The path tree after extraction:
TODO.

In order to train model yourself:
```bash
CUDA_VISIBLE_DEVICES=0 python main_train.py -dataset_name cifar10 -model_name resnet 
```

Or using tmuxp, train for all model architectures and trainsets:
```bash
cd bash_scripts
tmuxp load train.yaml
```

## 2. Extract features and logtis:
```
cd src
CUDA_VISIBLE_DEVICES=0 python main.py -model densenet -trainset cifar10
```

Or using tmuxp
```bash
cd bash_scripts
tmuxp load extract_features.yaml
```

The tree path after executing for DenseNet and WideResNet:
TODO

## 3. Compute In-Dist/OutOfDist score
The tree path after the score computation:
TODO

Or using tmuxp
```bash
cd bash_scripts
tmuxp load compute_score.yaml
```


## 4. Calculate performance
Using notebook
TODO

