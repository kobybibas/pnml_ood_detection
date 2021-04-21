# OOD Detection

Install requirements:
```bash
while read requirement; conda install --yes $requirement;or pip install $requirement; end < requirements.txt
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