# OOD Detection

Install requirements:
```bash
while read requirement; conda install --yes $requirement;or pip install $requirement; end < requirements.txt
```

```bash
# Downlaod OOD data
cd data
chmod 777 ./download_data.sh
./download_data.sh

# Download pretrained models
cd models
chmod 777 ./download_models.sh
./download_models.sh
```



## Extract features and logtis

Using the pretrained models, save the feautres of the last layer and the model outputs logits.

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py model=densenet trainset=svhn ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py model=resnet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py model=resnet trainset=svhn ;
```

ODIN:

```bash
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_odin model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_odin model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_odin model=densenet trainset=svhn ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_odin model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_odin model=resnet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_odin model=resnet trainset=svhn ;
```

Gram:

```bash
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_gram model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_gram model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_gram model=densenet trainset=svhn ;
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_gram model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_gram model=resnet trainset=cifar100 ;
CUDA_VISIBLE_DEVICES=0 python main_execute_method.py --config-name=extract_gram model=resnet trainset=svhn ;
```

Run comparison: in main_score.py fill the output folder of the previous scripts. Then run the following:

```bash
python main_score.py
```



## Optimize ODIN var

Script to optimize odin pertubation and temeprature parameters.

```bash
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=densenet trainset=svhn ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=resnet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=resnet trainset=svhn ;
```

```bash
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=densenet trainset=cifar10 is_with_pnml=true ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=densenet trainset=cifar100 is_with_pnml=true ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=densenet trainset=svhn is_with_pnml=true ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=resnet trainset=cifar10 is_with_pnml=true ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=resnet trainset=cifar100 is_with_pnml=true ;\
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py model=resnet trainset=svhn is_with_pnml=true ;
```

Reprudcing result using tmuxp (https://github.com/tmux-python/tmuxp):

```bsh
cd configs
tmuxp load tmuxp_main_optimize_odin.yaml
```