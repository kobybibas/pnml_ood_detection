# OOD Detection
Load conda environment
```bash
conda env create -f environment.yml
```
or install requirements:
```bash
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt 
```

```bash
# Download OOD data
cd bash_scripts
chmod 777 ./download_data.sh
./download_data.sh

# Download pretrained models
chmod 777 ./download_models.sh
./download_models.sh
```

### Download imagenet30
Follow https://github.com/alinlab/CSI

Imagenet30 training set:
https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view

Imagenet30 testing set:
https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view

Put and untar under ./data/Imagenet30
```
.
├── README.md
├── data
│   ├── Imagenet30
│   │   ├── one_class_test
│   │   ├── one_class_test.tar
│   │   ├── one_class_train
│   │   └── one_class_train.tar
```

## Execute methods

Using the pretrained models, score ood detection

```bash
cd bash_scripts
chmod 777 ./execute_methods.sh
./execute_methods.sh
```

Create paper's tables
```bash
cd src
python main main_create_tables.py
```

