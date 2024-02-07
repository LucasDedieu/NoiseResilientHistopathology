# NoiseResilientHistopathology
Official implementation for the paper "Contrastive-Based Deep Embeddings for Noise-Resilient Histopathology Image Classification".

## Requirements
```console
pip install -r requirements.txt
```

Please note: For embedding extractions, CTranspath and Pathoduet require two different versions of timm. Make sure you have the appropriate version for each of them.

## How To Run
##### Image Datasets
We use pythorch ImageFolder to load dataset so organize your datasets as:
```console
dataset
└──train
    └──class1
        └──img1.png
        └──img2.png
        ...
    └──class2
    ...
└──val
└──test
```
val and test folders are optional and can be obtained by splitting train.

##### Embedding Datasets
If you have manually extracted embeddings, place them in ```./deep_features/{dataset_name}```. Respect nomenclature : ```{train/val/test}_df_{backbone_name}.npy and {train/val/test}_labels_{backbone_name}.npy```.

Else, the code will extract embeddings using image dataset and place them in the right folder before training.

##### Configs for the experiment settings
Add '*.yaml' file in the config folder for each experiment.

Example: for CTransPath runs on NCT-CRC-HE-100k dataset, write config in ```configs/crc/ctranspath.yml```
```console
epochs: 300
patience: 20

dataset:
  name: 'crc'
  type: 'deep_features'
  backbone: 'ctranspath'
  batch_size: 512
  sigma: 0.4
  num_workers: 2
  val_size: 0.2
  
optimizer:
  name: SGD
  lr: 0.025
  weight_decay: 1.e-4
  momentum: 0.9

scheduler:
  name: CosineAnnealingLR
  T_max: $epochs

criterion:
  name: CrossEntropyLoss

warmup:
    active: False
```

##### Arguments
* noise_rate: noise rate
* config_path: path to the configs folder
* version: the config file name
* exp_name: name of the experiments (as note)
* seed: random seed
* resize: resize size (for image-based method)

Example for CTransPath runs on all noise rates for NCT-CRC-HE-100k dataset 
```console
#!/bin/bash
python_script="main.py"
seeds=(42 123 1999 78)
runs=("seed=42" "seed=123" "seed=1999" "seed=78")
noise_rates=("0." "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

for noise_rate in "${noise_rates[@]}"; do
    for i in $(seq 4 $END); do 
        echo "Running $python_script with noise_rate=$noise_rate and seed=${seeds[$i-1]}"
        python3 $python_script --config_path configs/crc/ --version ctranspath --exp_name ${runs[$i-1]} --noise_rate $noise_rate --seed ${seeds[$i-1]}
    done
done
```


##### Results and Logs
After running, results and logs should be in ```experiments/{dataset_name}/{version_name}/{noise_rate}/{exp_name}```


## Citing this work
If you use this code in your work, please cite the accompanying paper:

```

```
