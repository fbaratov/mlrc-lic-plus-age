#!/bin/bash

#This file is responsible for running all the models with defined configuration
#Use arguments to control the behaviour of this script.

# Set some default values for the arguments of the python scripts.
captions='human'
model='bert'
data='race'
epochs=5
learning_rate=1e-5
check=false
save_every=5
save_model=true

usage()
{
  echo "Usage: run_models 
                        [--captions TEXT]
                        Set the captions to use for training the model.
                        Select 'generated' or 'human'
                        
                        [--model TEXT]
                        Set the model to use for training.
                        Select 'lstm' or 'bert' or 'bert_pretrained'

                        [--data TEXT]
                        Set the dataset to use for training.
                        Select 'gender' or 'race' or 'age'

                        [--epochs NUMBER ]
                        Set the number of epochs you want
                        
                        [--learning_rate NUMBER ]
                        Set the learning rate 

                        [--save_model]
                        Set the flag for setting the model

                        [--every NUMBER]  
                        Indicate the saving frequency

                        [--check]
                        Indicates if you only wanna run check.
                        " 
  exit 2
}

PARSED_ARGUMENTS=$(getopt -a -n run_models -o '' --long captions:,model:,data:,epochs:,learning_rate:,save_model,every:,check -- "$@")
eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    --captions)   captions="$2"   ; shift 2   ;;
    --model)    model="$2"    ; shift 2  ;;
    --data) data="$2";  shift 2 ;;
    --epochs)   epochs="$2"   ; shift 2 ;;
    --learning_rate) learning_rate="$2"; shift 2;;
    --save_model) save_model=true; shift;;
    --every) save_every="$2"; shift 2;;
    --check) check=true epochs=1; shift;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

echo -e "\nRunning the model\n"
echo -e "##########################################"
echo -e "\tcaptions: $captions\n\tmodel: $model\n\tdata: $data\n\tepochs: $epochs\n\tlearning_rate: $learning_rate\n\tcheck:$check\n\tsave_model:$save_model\n\tsave_every:$save_every\n\t"
echo -e "##########################################"


#This script is responsible for running all the models with different seeds. 
#We still need to find a way to write the output to file.
models=('nic' 'sat' 'fc' 'att2in' 'updn' 'transformer' 'oscar' 'nic_plus' 'nic_equalizer')
#Run the models with only three seeds due to computational resources.
seeds=(0 12 456)

# We set calc_model_leak when data is generated.
# Otherwise calc_ann_leak
calc=calc_model_leak 
freeze_bert=false
if [ $captions == "human" ] 
  then
    calc=calc_ann_leak
fi
if [ $model != "bert_pretrained" ] 
  then
    echo "$data, $model"
    python_file=${data}_${model}_leakage.py
    echo ""
else
    echo "$data"
    python_file=${data}_bert_leakage.py
    freeze_bert=true   
fi

for m in ${models[@]}; do
    for seed in ${seeds[@]}; do
        if [ "$freeze_bert" = true ]; then
          echo "python3 $python_file --seed $seed --num_epochs $epochs --$calc True --cap_model $m  --learning_rate $learning_rate --freeze_bert $freeze_bert --save_model $save_model --every $save_every"
          python3 $python_file --seed $seed --num_epochs $epochs --$calc True --cap_model $m  --learning_rate $learning_rate --freeze_bert $freeze_bert --save_model $save_model --every $save_every
        else
          echo "python3 $python_file --seed $seed --num_epochs $epochs --$calc True --cap_model $m  --learning_rate $learning_rate --save_model $save_model --every $save_every"   
          python3 $python_file --seed $seed --num_epochs $epochs --$calc True --cap_model $m  --learning_rate $learning_rate --save_model $save_model --every $save_every
        fi
    done
done


exit 0;
