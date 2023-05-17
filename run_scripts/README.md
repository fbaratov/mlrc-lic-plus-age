<h1> Usage </h1>

To use the script, you need to be in the root directory of the repository. If you are not in there, then you will get a FileNotFound error.

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
                        
If the `--check` flag is used, all the models are run for 1 epochs only.

<h2> Using the script for training different models </h2>

There are 3 different models that we train on 2 different datasets.

<h3> LSTM </h3>

Use the script with the values defined in the paper for LSTM.
This is how you can compute the LIC values for the LSTM model.

```bash
bash run_models.sh --model lstm --captions human --data gender --epochs 20 --learning_rate 5e-5 > outputs/lstm_human_gender.txt
bash run_models.sh --model lstm --captions generated --data gender --epochs 20 --learning_rate 5e-5 > outputs/lstm_generated_gender.txt
cd outputs
python3 ./scripts/calculate_LIC.py
```



<h3> BERT-ft </h3>

Use the script with the values defined in the paper for BERT-ft.
This is how you can compute the LIC values for BERT-ft.

```bash
bash run_models.sh --model bert --captions human --data gender --epochs 5 --learning_rate 5e-5 > outputs/bert_human_gender.txt
bash run_models.sh --model bert --captions generated --data gender --epochs 5 --learning_rate 5e-5 > outputs/bert_generated_gender.txt
cd outputs
python3 ./scripts/calculate_LIC.py
```



<h3> BERT-pre </h3>

Use the script with the values defined in the paper for BERT-pre.
This is how you can compute the LIC values for BERT-pre.

```bash
bash run_models.sh --model bert_pretrained --captions human --data gender --epochs 20 --learning_rate 5e-5 > outputs/pretrained_bert_human_gender.txt
bash run_models.sh --model bert_pretrained --captions generated --data gender --epochs 20 --learning_rate 5e-5 > outputs/pretrained_bert_generated_gender.txt
cd outputs
python3 ./scripts/calculate_LIC.py
```
