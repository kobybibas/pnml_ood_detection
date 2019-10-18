# OOD Detection

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


In order to train model yourself:
```bash
CUDA_VISIBLE_DEVICES=0 python main_train.py -model_name densenet -dataset_name cifar10 
```

Or using tmuxp, train for all model architectures and trainsets:
```bash
cd bash_scripts
tmuxp load train.yaml
```

## 2. Extract features and logtis:
Using the pretrained model, save the feautres of the last layer and the model outputs logits.
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
```
.
├── README.md
├── bash_scripts
│   ├── extract.yaml
│   ├── score.yaml
│   └── train.yaml
├── data
├── models
├── notebooks
├── output
│   ├── features
│   ├── logits
├── requirements.txt
└── src
```

## 3. Compute In-Dist/OutOfDist score
Compute the score from which one can determine if the test sample is In or Out of Distribution.

```bash
cd bash_scripts
CUDA_VISIBLE_DEVICES=0 python  main_score.py -model densenet -dataset cifar10 
```
Or using tmuxp
```bash
cd bash_scripts
tmuxp load compute_score.yaml
```

The tree path after the score computation:
```
.
├── README.md
├── bash_scripts
│   ├── extract.yaml
│   ├── score.yaml
│   └── train.yaml
├── data
├── models
├── notebooks
│   ├── benchmark.ipynb
│   └── plot_functions.ipynb
├── output
│   ├── features
│   ├── logits
│   ├── score
├── requirements.txt
└── src
```

## 4. Calculate performance
Using the notebook
[./notebooks/ benchmark.ipynb](./notebooks/ benchmark.ipynb)


