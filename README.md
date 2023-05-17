_Note: The code here is an extension and reproduction of the research done by [Hirota et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Hirota_Quantifying_Societal_Bias_Amplification_in_Image_Captioning_CVPR_2022_paper.html). This paper's code is available in their [GitHub repository](https://github.com/rebnej/lick-caption-bias). Much of the scripts and code available in this project are taken directly from the aforementioned repository._


# Reproducing Our Results

Our repository is structured as follows.

* The root directory contains the bulk of our code, as well as notebooks that we used for running the models in Google Colab.
    
* [./Dockerfiles](https://github.com/fbaratov/mlrc-lic-plus-age/tree/main/Dockerfiles) contains Docker files for building Docker images. Please refer to [Running Docker Files](#running-docker-files) for a description of how to use them.

* [./bias_data](https://github.com/fbaratov/mlrc-lic-plus-age/tree/main/bias_data) contains the human annotated and generated captions datasets for all models.

* [./outputs](https://github.com/fbaratov/mlrc-lic-plus-age/tree/main/outputs) contains the outputs of our model runs, as well as the scripts used to parse them. This way, we can have an
easier time verifying results. The scripts need to be executed in the project root directory. For more information, please refer to [Parsing Results](#parsing-results).

* [./run_scripts](https://github.com/fbaratov/mlrc-lic-plus-age/tree/main/run_scripts) contains a bash file to run all models with specified seeds on partition of dataset 'gender' 'race' or 'age'.
For more information, please refer to [Running Models](#running-models).

# Quick run in Colab
The simplest way to run the models is in Google Colab:

1. Use either ```run_bert.ipynb``` or ```run_lstm.ipynb``` as the main notebook file.
2. Upload or clone the repository to the runtime.
3. Run the notebook fully. It will install the environment and provide all relevant results for the respective model.

# Running Models

‼️ Firstly the environment dependencies needs to be installed. 

We trained BERT-ft and BERT-pre models using the LISA computational cluster, as their computational overhead is much less significant than that of LSTM.
We trained LSTMs using Google Colab. However, you are free to use whatever resources you have available :)

### Local installation

First of all, make sure you are running Python 3.7. Then, run the following based on which model you want to use:

<h3> BERT-ft and BERT-pre </h3>

``` 
python3 -m pip install -r bert_requirements.txt 
```

<h3> LSTM </h3>

```
python3 -m pip install -r lstm_requirements.txt
```

Please use separate environments for BERT and LSTM requirements, as using them together may result in errors and conflicts. 

Once the environment is set up, the models are ready to go. We adapted a bash script to run the models with one command. 

```
bash run_models.sh --model lstm --captions human --data age --epochs 20 --learning_rate 5e-5 > outputs/lstm_human_age.txt
```

This command runs the lstm model with age data using human captions, thereby producing the $LIC_D$. The output of the bash script is then
written in a concise manner into the [./outputs](https://github.com/fbaratov/mlrc-lic-plus-age/tree/main/outputs) directory.


For more information about the bashscript please refer to [run_scripts/README.md](https://github.com/fbaratov/mlrc-lic-plus-age/blob/main/run_scripts/README.md).


‼️ When saving the output of this bash script use this template: 

```
outputs/model_captions_data.txt
```

‼️ This bash script needs to be executed from the project root directory, otherwise it will throw a FileNotFound error.

# Parsing Results

We have developed a script to parse .txt file from the run, into Python class.
Calling the function `python3 calculate_LIC.py` parses the text files and produces results for us to verify and use.
For verification, you can check the epochs, seeds, model names and observe if it is as expected.
For result generation, you can check the mean and standard deviation of $LIC_M$ and $LIC_D$ results, as well as the resultant $LIC$, which is equal to $LIC_M - LIC_D$.


:bangbang: This script needs to be run in the project's root directory.

:heavy_check_mark: This script is tested using [./outputs/tests](https://github.com/fbaratov/mlrc-lic-plus-age/tree/main/outputs/tests)

# Running docker files

First of all make sure to install docker :) 

Follow the directory to the project's root directory, then execute this command.

<h3> LSTM </h3>

```
docker build -f ./Dockerfiles/Dockerfile_lstm -t lstm. 
docker run -it lstm /bin/bash
#In the container
cd mlrc-lic-plus-age
```

<h3> BERT-ft and BERT-pre </h3>

```
docker build -f ./Dockerfiles/Dockerfile_bert -t bert. 
docker run -it bert_fact /bin/bash
#In the container
cd mlrc-lic-plus-age
```

You should be in the container now. Go to the project's root directory and run the scripts as described below. 
