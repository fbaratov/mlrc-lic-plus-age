# This bash script is responsible for the race lstm model and generated captions.
# We observe the output of the model, and if there is not errro in 1 epoch, 
# then we move to next one.


#!/bin/bash

# Initialize the array of models
models=('nic' 'sat' 'fc' 'att2in' 'updn' 'transformer' 'oscar' 'nic_equalizer' 'nic_plus')

# For each model run the python script for lstm leakage.
for model in ${models[@]}; do
    echo "----------------------------"
    echo -e "Checking:race_lstm_leakage.py \ncal_model_leak: True \nmodel: $model \nseed: 0"
    echo "----------------------------"
    # We calculate race_lstm_leakeage with model_leak for generated captions.
    python3 race_lstm_leakage.py --seed 0 --num_epochs 1 --calc_model_leak True --cap_model $model
done