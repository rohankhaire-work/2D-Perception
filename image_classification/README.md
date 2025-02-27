# PlantVillage-DeepLearning
Deep Learning solution for detecting diseases on leaves in the PlantVillage dataset.

In this project I try to gauge the effects of **transfer learning**, **hyper-parameter tuning**, and **quantization** for the sake of learning to implement production-level deployment.

# Data
The data used for the experiments can be found [here](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

Only the coloured data is used for image classification.

# Dependencies
The libraries are installed via **Poetry** and are based on **Python 3.10**.

# Installation
```bash
# clone the repo
git clone https://github.com/rohankhaire-work/PlantVillage-DeepLearning.git

# Install the libraries
cd PlantVillage-DeepLearning
poetry install

# Unzip the data in the data folder
cd data
unzip archive.zip
```

# Hyper-parameter tuning
```bash
# Currently lr, momentum, and batch size are tuned
poetry run python hyperparameter_tuning.py
```

# Training
```bash
# Train the networks
poetry run python main.py
```

# Results
| Model              | Test Accuracy |
| :---------------- | :-----------:  |
| Resnet50          |   94.47        |
| mobilenetv2       |   95.54        |
| custom_CNN        |   97.21        |
