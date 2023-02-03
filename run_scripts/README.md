<h1> Usage </h1>

To use the script, you need to be in the ~/mlrc-lic-plus-age directory of the repository. If you are not in there, then you will get an error saying file not found.

<h2> Arguments </h2>

The script accepts 6 arguments in total.

                        [--captions TEXT]
                        Set the captions to use for training the model.
                        Select 'generated' or 'human'
                        
                        [--model TEXT]
                        Set the model to use for training.
                        Select 'lstm' or 'bert' or 'bert_pretrained'
                        
                        [--data TEXT]
                        Set the dataset to use for training.
                        Select 'gender' or 'race'
                        
                        [--epochs NUMBER ]
                        Set the number of epochs you want
                        
                        [--learning_rate NUMBER ]
                        Set the learning rate 
                        
                        [--check]
                        Indicates if you only wanna run check.
                        
If --check flag is used, all the models are run for 1 epochs only.

<h2> Using the script for training different models </h2>

There are 3 different models that we need to train on 2 different dataset.

<h3> LSTM </h3>

Use the script with defiend values in the paper for lstm.
This is how you can compute the LIC values for lstm model.

```bash
bash run_models.sh --model lstm --captions human --data gender --epochs 20 --learning_rate 5e-5 > outputs/lstm_human_gender.txt
bash run_models.sh --model lstm --captions generated --data gender --epochs 20 --learning_rate 5e-5 > outputs/lstm_generated_gender.txt
cd outputs
python3 ./scripts/calculate_LIC.py
```



<h3> BERT </h3>

Use the script with defiend values in the paper for bert.
This is how you can compute the LIC values for bert model.

```bash
bash run_models.sh --model bert --captions human --data gender --epochs 5 --learning_rate 5e-5 > outputs/bert_human_gender.txt
bash run_models.sh --model bert --captions generated --data gender --epochs 5 --learning_rate 5e-5 > outputs/bert_generated_gender.txt
cd outputs
python3 ./scripts/calculate_LIC.py
```



<h3> Pretrained BERT </h3>

Use the script with defiend values in the paper for pre_trained bert.
This is how you can compute the LIC values for pre_trained bert model.

```bash
bash run_models.sh --model bert_pretrained --captions human --data gender --epochs 20 --learning_rate 5e-5 > outputs/pretrained_bert_human_gender.txt
bash run_models.sh --model bert_pretrained --captions generated --data gender --epochs 20 --learning_rate 5e-5 > outputs/pretrained_bert_generated_gender.txt
cd outputs
python3 ./scripts/calculate_LIC.py
```
