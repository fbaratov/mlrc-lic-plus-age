# This bash script is responsible for the gender lstm model and human captions
# We observe the output of the model, and if there is not errro in 1 epoch, 
# then we move to next one.


#!/bin/bash

# Initialize the array of models
models=('nic' 'sat' 'fc' 'att2in' 'updn' 'transformer' 'oscar' 'nic_equalizer' 'nic_plus')

# For each model run the python script for lstm leakage.
for model in ${models[@]}; do
    echo "----------------------------"
    echo -e "Checking:lstm_leakage.py \ncal_ann_leak: True \nmodel: $model \nseed: 0"
    echo "----------------------------"
    python3 lstm_leakage.py --seed 0 --num_epochs 1 --calc_ann_leak True --cap_model $model
done
