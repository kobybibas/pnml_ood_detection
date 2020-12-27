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
CUDA_VISIBLE_DEVICES=7 python main_extract.py model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=7 python main_extract.py model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=7 python main_extract.py model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=7 python main_extract.py model=resnet trainset=cifar100 ;
```

ODIN:

```bash
CUDA_VISIBLE_DEVICES=6 python main_extract.py --config-name=extract_odin model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=6 python main_extract.py --config-name=extract_odin model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=6 python main_extract.py --config-name=extract_odin model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=6 python main_extract.py --config-name=extract_odin model=resnet trainset=cifar100 ;
```




## Optimize ODIN var

```bash
CUDA_VISIBLE_DEVICES=7 python optimize_odin.py model=densenet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=7 python optimize_odin.py model=densenet trainset=cifar100 ;\
CUDA_VISIBLE_DEVICES=7 python optimize_odin.py model=resnet trainset=cifar10 ;\
CUDA_VISIBLE_DEVICES=7 python optimize_odin.py model=resnet trainset=cifar100 ;
```


