_Note: the repository of the paper that we are reproducing can be found [here](https://github.com/rebnej/lick-caption-bias). Much of the scripts and code available in this project are taken directly from the aforementioned repository._



# Reproducing Our Results

Our repository is structured as follows.

* The root directory contains the python scripts for runnig the models, as well as .job scripts for running the models in LISA cluster.
    
* ./Dockerfiles contains dockerfiles for building docker images. We have a lot of environmental requirements, so it is wise to use
this when you do not have access to LISA cluster not UNIX based system. Explanation of how to use docker is out of scope of this repository, 
but we provide commands for getting the models work in docker.[Running Docker Files](#running-docker-files)

* ./bias_data contains the human annotated and generated captions for each model. This is our entry point to the dataset.
    
* ./notebooks contains the notebook for running the models in google colab. It is not very important. 
    
* ./outputs contains the outputs of the model runs, also scripts to parse them into python classes. Therefore, we can have
easier time verifying them. The scripts needs to be executed in the project root directory. For more information about using this script please refer to 
the section [Parsing Results](#parsing-results)

* ./run_scripts contains a bash file to run all models with specified seeds on partition of dataset 'gender' 'race' or 'age'.

For more information about using the script please refer to the section [Running Models](#running-models).

# Quick run in Colab
The most out-of-box way to run the models is in Google Colab. The following steps apply:

1. Use either ```run_bert.ipynb``` or ```run_lstm.ipynb``` as the main notebook file.
2. Upload or clone the repository to the runtime.
3. Run the notebook fully. It will install the environment and provide all relevant results for the respective model.

# Running Models


‼️ Firstly the environment dependencies needs to be installed. 

We only train bert and pretrained bert models in LISA cluster because their computational overhead is not as significant as LSTM.

### Local installation

First of all, make sure you are running python 3.7.
Installing python 3.7 is out of scope for this readme, so it is assumed that you already installed it and it is working properly.

Depending on which model you want to use:

For pre_trained bert and fine tuned bert.
```
python3 -m pip install -r bert_requirements.txt
```

For LSTM.
```
python3 -m pip install -r lstm_requirements.txt
```

Now, you have the environment set up. And ready to run the bash script.


We adapted a bash script to run the models with one command. 

```
bash run_models.sh --model lstm --captions human --data age --epochs 20 --learning_rate 5e-5 > outputs/lstm_human_age.txt
#Run the lstm model with age data using human captions. This produces the LIC_D for lstm model. The output of the bash script needs to be
written in concside manner to outputs directory.
```

For more information about the bashscript please refer to run_scripts/README.md .
file.


‼️ When saving the output of this bash script use this template: 

```
outputs/model_captions_data.txt
```

‼️ This bash script needs to be executed from the project root directory, otherwise it will throw file not found error.

# Parsing Results

We have developed a script to parse .txt file from the run, into python class.
Calling the function python3 calculate_LIC.py parses the text files and produces results for us to verify and use.
For verification, you can check the epochs, seeds, model names and observe if it is expected.
For result generation, you can use the mean and std of LIC_M and LIC_D results.
As well as LIC results. Which is basically LIC_M - LIC_D


:bangbang: This script needs to be run in the project's root directory.

:heavy_check_mark: This script is tested using ./outputs/test

# Running docker files

First of all make sure to install docker :) 

Follow the directory to the lick-caption-bias, and when you are in the directory execute this command.

<h3> For lstm </h3>

```
docker build -f ./Dockerfiles/Dockerfile_lstm -t lstm. 
docker run -it lstm /bin/bash
#In the container
cd lick-caption-bias
```

<h3> For BERT </h3>

```
docker build -f ./Dockerfiles/Dockerfile_bert -t bert. 
docker run -it bert_fact /bin/bash
#In the container
cd lick-caption-bias
```

You should be in the container now. Go to the lick-caption-bias directory and run the scripts as described below. 
