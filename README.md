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

