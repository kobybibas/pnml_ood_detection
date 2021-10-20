# Single Layer Predictive Normalized Maximum Likelihood for Out-of-Distribution Detection 
https://arxiv.org/abs/2110.09246

This is a fast and scalable approch for detecting out-of-dsitribution test samples.
It can be applied to any pretrained model.



# Pseudocode
```python
# Assuming to have:trainloader, testloader, model with model.backbone and model.classifer methods

# Extract the features of the training set. Dimensions: training_size x num_features
features = torch.vstack([model.backbone(images) for images, label in trainloader])

# Compute the empirical correlation matrix inverse
x_t_x_inv = torch.linalg.inv(features.T @ features)

# Calculate the regret: Large regret means out-of-distribution sample
for images, labels in testloader:
    features = model.backbone(images)
    probs = torch.softmax(model.classifier(features), dim=-1)

    x_proj = features @ x_t_x_inv @ features.T
    xt_g = x_proj / (1 + x_proj)

  # Equation 20
    regrets = torch.sum(probs / (probs + (1 - probs) * probs ** xt_g),
                        dim=-1)
```

# Paper results



# Run to code

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

