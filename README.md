# NoiseResilientHistopathology
Official implementation for the paper "Contrastive-Based Deep Embeddings for Noise-Resilient Histopathology Image Classification".

## Requirements
```console
pip install -r requirements.txt
```

## How To Run
##### Configs for the experiment settings
Check '*.yaml' file in the config folder for each experiment.

##### Arguments
* noise_rate: noise rate
* asym: use if it is asymmetric noise, default is symmetric
* config_path: path to the configs folder
* version: the config file name
* exp_name: name of the experiments (as note)
* seed: random seed

Example for baseline runs on all noise rates for NCT-CRC-HE-100k dataset 
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


## Citing this work
If you use this code in your work, please cite the accompanying paper:

```

```
