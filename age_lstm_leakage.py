"""
gender_lstm_leakage.py, but modified to be used with age-based captions.
"""

import pickle
import pandas as pd
import random
from collections import namedtuple
import torchtext
import torch
import csv
import spacy
import re
from torchtext.legacy import data
import pickle
import random
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import time
import argparse
import numpy as np
import os
import pprint
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# import unchanged functions from original file, just to be clear what is modified and what isn't
from gender_lstm_leakage import binary_accuracy, RNN, count_parameters, train

def gender_pickle_generator(cap_model):
  '''
    This function generates the captions for the specified model. It only concerns
    age. It could be merged, but for the purposes of readability it is better to 
    keep this way.
    Arguments
    ---------
    cap_model : str
        The captioning model in string format.
  '''
  directory_age = {'nic':  pickle.load(open('bias_data/Show-Tell/gender_val_st10_th10_cap_mw_entries.pkl', 'rb')),
             'sat':         pickle.load(open('bias_data/Show-Attend-Tell/gender_val_sat_cap_mw_entries.pkl', 'rb')),
             'fc' :         pickle.load(open('bias_data/Att2in_FC/gender_val_fc_cap_mw_entries.pkl', 'rb')),
             'att2in':      pickle.load(open('bias_data/Att2in_FC/gender_val_att2in_cap_mw_entries.pkl', 'rb')),
             'updn':        pickle.load(open('bias_data/UpDn/gender_val_updn_cap_mw_entries.pkl', 'rb')),
             'transformer': pickle.load(open('bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl', 'rb')),
             'oscar':       pickle.load(open('bias_data/Oscar/gender_val_cider_oscar_cap_mw_entries.pkl', 'rb')),
             'nic_plus':    pickle.load(open('bias_data/Woman-Snowboard/gender_val_baselineft_cap_mw_entries.pkl', 'rb')),
             'nic_equalizer': pickle.load(open('bias_data/Woman-Snowboard/gender_val_snowboard_cap_mw_entries.pkl', 'rb')),
             'human':         pickle.load(open('bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl', 'rb'))                          
             }
  return directory_age[cap_model]
       
def race_pickle_generator(cap_model):
  '''
    This function generates the captions for the specified model. It only concerns
    race. It could be merged, but for the purposes of readability it is better to 
    keep this way.
    Arguments
    ---------
    cap_model : str
        The captioning model in string format.
  '''
  directory_race = {'nic': pickle.load(open('bias_data/Show-Tell/race_val_st10_cap_entries.pkl', 'rb')),
             'sat':         pickle.load(open('bias_data/Show-Attend-Tell/race_val_sat_cap_entries.pkl', 'rb')),
             'fc' :         pickle.load(open('bias_data/Att2in_FC/race_val_fc_cap_entries.pkl', 'rb')),
             'att2in':      pickle.load(open('bias_data/Att2in_FC/race_val_att2in_cap_entries.pkl', 'rb')),
             'updn':        pickle.load(open('bias_data/UpDn/race_val_updn_cap_entries.pkl', 'rb')),
             'transformer': pickle.load(open('bias_data/Transformer/race_val_transformer_cap_entries.pkl', 'rb')),
             'oscar':       pickle.load(open('bias_data/Oscar/race_val_cider_oscar_cap_entries.pkl', 'rb')),
             'nic_plus':    pickle.load(open('bias_data/Woman-Snowboard/race_val_baselineft_cap_entries.pkl', 'rb')),
             'nic_equalizer': pickle.load(open('bias_data/Woman-Snowboard/race_val_snowboard_cap_entries.pkl', 'rb')),
             'human':         pickle.load(open('bias_data/Human_Ann/race_val_obj_cap_entries.pkl', 'rb')),                         
             }
  return directory_race[cap_model]

def label_human_caption(caption_list, young_words, old_words):
    '''
    Label human caption list from one entry of human captions.
    This rule is applied if any of the caption in the list contains 
    any word in young_words and old_words.
    The logic rules are simple:
    If entry corresponds to both old and young it is probably young 
    as we include man and woman in the words
    If entry is only young then it is young
    If entry is only old then it is old.
    Arguments
    ---------
    caption_list : list[str]
        The list of captions from five different human annotators
    young_words : list[str]   
        The list of words that is specified as young
    old_words : list[str]
        The list of words that is specified as old. 
    '''
    # If any of the words from any of the caption in the old_words list
    exists_old = any(word in caption.split() for word in old_words for caption in caption_list)
    # If any of the words from any of the caption is in young_words list
    exists_young = any(word in caption.split() for word in young_words for caption in caption_list)
    if exists_old and exists_young:
      return "Young"
    if exists_old:
        return "Old"
    if exists_young:
        return "Young"

def label_human_annotations(captions, young_words, old_words):
  '''
  This functions labels the human annotations from the captions dictionary.
  The captions dictionary is from their data.
  The dictionary has 5 different captions for a unique image id.
  If a caption contains the young or old words, then we label it as such.
  Arguments
  ---------
  captions : list[dict]
      Captions from the mscoco dataset
  young_words : list[str]
      List of words representing the young words
  old_words : list[str]
      List of words representing the old words
  '''
  # Create a data frame from captions
  data_frame = pd.DataFrame.from_records(captions,index=[str(id) for id in range(len(captions))])
  # Label the ages using the label_human_caption function
  labelled_ages = [label_human_caption(list, young_words, old_words) for list in data_frame['caption_list']]
  data_frame.insert(2, "bb_age", labelled_ages, True)
  data_frame['img_id'] =data_frame['img_id'].astype(int)
  return data_frame

def match_labels(labelled_human_captions, generated_captions):
  '''
  This function maps the labelled human captions to generated captions
  So we have the labels for generated captions as well.
  Arguments
  ---------
  labelled_human_captions : list[dict]
      labelled human captions dictionary
  generated_captions : list[dict]
      generated captions from the specified model.
  '''
  generated_captions_data_frame = pd.DataFrame.from_records(generated_captions,index=[str(id) for id in range(len(generated_captions))])
  generated_captions_data_frame['img_id'] = generated_captions_data_frame['img_id'].astype(int)
  # Merge the datasets using the img_id column because it has one to one match.
  generated_captions_data_frame = generated_captions_data_frame.merge(labelled_human_captions[['img_id', 'bb_age']], on='img_id')
  return generated_captions_data_frame

def make_train_test_split(args, age_task_mw_entries):
    '''
    This is the function from the authors code that define train and test splits
    from the entries.

    Arguments
    ---------
    args : NamedTuple
        The parsed arguments
    age_task_entries : list[dict]
        The entries for age task. This is basically the labelled dataframe.
    '''
    if True:
        old_entries, young_entries = [], []
        for _ , entry in age_task_mw_entries.iterrows():
            if entry['bb_age'] == 'Young':
                young_entries.append(entry)
            else:
                old_entries.append(entry)
        #print(len(old_entries))
        each_test_sample_num = round(len(young_entries) * args.test_ratio)
        each_train_sample_num = len(young_entries) - each_test_sample_num

        old_train_entries = [old_entries.pop(random.randrange(len(old_entries))) for _ in range(each_train_sample_num)]
        young_train_entries = [young_entries.pop(random.randrange(len(young_entries))) for _ in range(each_train_sample_num)]
        old_test_entries = [old_entries.pop(random.randrange(len(old_entries))) for _ in range(each_test_sample_num)]
        young_test_entries = [young_entries.pop(random.randrange(len(young_entries))) for _ in range(each_test_sample_num)]
        d_train = old_train_entries + young_train_entries
        d_test = old_test_entries + young_test_entries
        random.shuffle(d_train)
        random.shuffle(d_test)
        print('#train : #test = ', len(d_train), len(d_test))

    return d_train, d_test

def get_parser():
    '''
    Returns arguments parser. Barely modified, the only difference is that no "gender_or_race" arg is present.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)

    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_age_words", default=True, type=bool)
    parser.add_argument("--save_preds", default=False, type=bool)
    parser.add_argument("--use_glove", default=False, type=bool)
    parser.add_argument("--save_model_vocab", default=False, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)
    parser.add_argument("--mask_bias_source", default='', type=str, help='obj or person or both or none')

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--embedding_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--pad_idx", default=0, type=int)
    parser.add_argument("--fix_length", default=False, type=bool)

    return parser


def evaluate(model, iterator, criterion, batch_size, TEXT, args):

    calc_score = True
    calc_mw_acc = True 

    m = nn.Sigmoid()
    total_score = 0

    epoch_loss = 0
    epoch_acc = 0

    young_preds_all, old_preds_all = list(), list()
    young_scores_all, old_scores_all = list(), list()
    young_truth_all, old_truth_all = list(), list()
    all_pred_entries = []

    model.eval()
    
    with torch.no_grad():

        cnt_data = 0
        for i, batch in enumerate(iterator):

            text, text_lengths = batch.prediction

            predictions, _ = model(text, text_lengths)
            predictions = predictions.squeeze(1)
            cnt_data += predictions.size(0) 

            loss = criterion(predictions, batch.label.to(torch.float32))

            acc = binary_accuracy(predictions, batch.label.to(torch.float32))

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if calc_score:
                probs = m(predictions).cpu() #[batch_size]
                pred_ages = (probs >= 0.5000).int()

                correct = torch.eq(pred_ages, batch.label.to(torch.int32).cpu())

                pred_score_tensor = torch.zeros_like(correct, dtype=float)

                for i in range(pred_score_tensor.size(0)):
                    old_score = probs[i]
                    young_score = 1 - old_score
                    if young_score >= old_score:
                        pred_score = young_score
                    else:
                        pred_score = old_score

                    pred_score_tensor[i] = pred_score

                scores_tensor = correct.int() * pred_score_tensor
                correct_score_sum = torch.sum(scores_tensor)
                total_score += correct_score_sum.item()


            if calc_mw_acc:
                probs = m(predictions).cpu() #[batch_size]
                pred_ages = (probs >= 0.5000).int()
                young_target_ind = [i for i, x in enumerate(batch.label.to(torch.int32).cpu().numpy().tolist()) if x == 0]
                old_target_ind = [i for i, x in enumerate(batch.label.to(torch.int32).cpu().numpy().tolist()) if x == 1]
                young_pred = [*itemgetter(*young_target_ind)(pred_ages.tolist())]
                old_pred = [*itemgetter(*old_target_ind)(pred_ages.tolist())]
                young_scores = [*itemgetter(*young_target_ind)(probs.tolist())]
                young_scores = (1 - torch.tensor(young_scores)).tolist()
                old_scores = [*itemgetter(*old_target_ind)(probs.tolist())]
                young_target = [*itemgetter(*young_target_ind)(batch.label.to(torch.int32).cpu().numpy().tolist())]
                old_target = [*itemgetter(*old_target_ind)(batch.label.to(torch.int32).cpu().numpy().tolist())]
                young_preds_all += young_pred
                young_scores_all += young_scores
                young_truth_all += young_target
                old_preds_all += old_pred
                old_scores_all += old_scores
                old_truth_all += old_target
                
            if args.save_preds:
                probs = m(predictions).cpu() #[batch_size]
                pred_ages = (probs >= 0.5000).int()

                for i, (imid, fs, pg) in enumerate(zip(batch.imid, probs, pred_ages)):
                    image_id = imid.item()
                    old_score = fs.item()
                    young_score = 1 - old_score
                    
                    sent_ind = text[:, i]
                    sent_list = []
                    for ind in sent_ind:
                        word = TEXT.vocab.itos[ind]
                        sent_list.append(word)
                    sent = ' '.join([c for c in sent_list])

                    all_pred_entries.append({'image_id':image_id, 'young_score':young_score, 'old_score':old_score, 'input_sent': sent})


    if calc_mw_acc:
        young_acc = accuracy_score(young_truth_all, young_preds_all)
        old_acc = accuracy_score(old_truth_all, old_preds_all)
        young_correct = torch.eq(torch.tensor(young_preds_all), torch.tensor(young_truth_all))
        old_correct = torch.eq(torch.tensor(old_preds_all), torch.tensor(old_truth_all))
        young_scores_tensor = young_correct.int() * torch.tensor(young_scores_all)
        young_score_sum = torch.sum(young_scores_tensor).item()
        young_score_avg = young_score_sum / (len(iterator) * batch_size * 0.5)
        old_scores_tensor = old_correct.int() * torch.tensor(old_scores_all)
        old_score_sum = torch.sum(old_scores_tensor).item()
        old_score_avg = old_score_sum / (len(iterator) * batch_size * 0.5) 
    else:
        young_acc, old_acc = None, None
        

    if args.save_preds:
        file_name = '%s_pred_entries_seed%s.pkl' %(args.cap_model, args.seed)
        save_path = os.path.join('/bias-vl/LSTM', args.cap_model, file_name )
        pickle.dump(all_pred_entries, open(save_path, 'wb'))


    return epoch_loss / len(iterator), epoch_acc / len(iterator), total_score / cnt_data, young_acc, old_acc, young_score_avg, old_score_avg



def main(args):
    '''
    Modified version of main from age_lstm_leakage.

    Arguments
    ---------
    args : NamedTuple
        The parsed arguments. See get_parser() for more details.
    '''
    # remove previously generated train/val/test files
    if os.path.exists('bias_data/train.csv'):
        os.remove('bias_data/train.csv')
    if os.path.exists('bias_data/val.csv'):
        os.remove('bias_data/val.csv')
    if os.path.exists('bias_data/test.csv'):
        os.remove('bias_data/test.csv')

    # set up seeds, gpu learning
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    
    TEXT = data.Field(tokenize = 'spacy', tokenizer_language ='en_core_web_sm', include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)
    young_words = ['kid', 'kids', 'child', 'children', 'young', 'boy', 'boys', 'little', 'baby', 'babies',
               'childhood', 'babyhood', 'toddler', 'adolescence', 'adolescent', 'teenager', 'teenagers',
               'schoolboy', 'schoolgirl', 'youngster', 'infant', 'preschooler'
               'toddler', 'student']
    old_words = ['elder', 'man', 'men', 'woman', 'women', 'old', 'elders', 'elderly', 'grandma', 'grandpa',
             'mom', 'dad', 'father', 'ancient', 'elder', 'aged', 'senior',
             'grandparent', 'senior']
    age_words = young_words + old_words  
    age_val_obj_cap_entries = label_human_annotations(gender_pickle_generator('human'),young_words,old_words) # Human captions
    selected_cap_age_entries = match_labels(age_val_obj_cap_entries,gender_pickle_generator(args.cap_model))
    print("Generated using {}".format(args.cap_model))

    ##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN Leakage ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- task is Captioning --')
            d_train, d_test = make_train_test_split(args, age_obj_cap_mw_entries)
            
            val_acc_list = []
            young_acc_list, old_acc_list = [], [] 
            score_list = []
            young_score_list, old_score_list = [], []
            rand_score_list = []
            
            if args.align_vocab:
                model_vocab = pickle.load(open('bias_data/model_vocab/%s_vocab.pkl' %args.cap_model, 'rb'))
                print('len(model_vocab):', len(model_vocab))

            for cap_ind in range(5):
                if args.mask_age_words:
                    with open('bias_data/train.csv', 'w') as f:
                        writer = csv.writer(f)
                        for i, entry in enumerate(d_train):
                            if entry['bb_age'] == 'Young':
                                age = 0
                            else:
                                age = 1
                            ctokens = word_tokenize(entry['caption_list'][cap_ind].lower())
                            new_list = []
                            for t in ctokens:
                                if t in age_words:
                                    new_list.append('ageword')
                                elif args.align_vocab:
                                    if t not in model_vocab:
                                        new_list.append('<unk>')
                                    else:
                                        new_list.append(t)      
                                else:
                                    new_list.append(t)

                            new_sent = ' '.join([c for c in new_list])
                            if i <= 10 and cap_ind == 0 and args.seed == 0:
                                print(new_sent)

                            writer.writerow([new_sent.strip(), age, entry['img_id']])

                    with open('bias_data/test.csv', 'w') as f:
                        writer = csv.writer(f)
                        for i, entry in enumerate(d_test):
                            if entry['bb_age'] == 'Young':
                                age = 0
                            else:
                                age = 1
                            ctokens = word_tokenize(entry['caption_list'][cap_ind].lower())
                            new_list = []
                            for t in ctokens:
                                if t in age_words:
                                    new_list.append('ageword')
                                elif args.align_vocab:
                                    if t not in model_vocab:
                                        new_list.append('<unk>')
                                    else:
                                        new_list.append(t)
                                else:
                                    new_list.append(t)

                            new_sent = ' '.join([c for c in new_list])

                            writer.writerow([new_sent.strip(), age, entry['img_id']])

                else:
                    print("!! SHOULD MASK AGE WORDS")
                    break


                nlp = spacy.load("en_core_web_sm")

                TEXT = data.Field(sequential=True, tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True, use_vocab=True)
                LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
                IMID = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

                train_val_fields = [
                    ('prediction', TEXT), # process it as text
                    ('label', LABEL), # process it as label
                    ('imid', IMID)
                    ]

                train_data, test_data = torchtext.legacy.data.TabularDataset.splits(path='bias_data/',train='train.csv', test='test.csv',
                                                                            format='csv', fields=train_val_fields)

                MAX_VOCAB_SIZE = 25000

                if args.use_glove:
                    TEXT.build_vocab(train_data, vectors = "glove.6B.100d",  max_size = MAX_VOCAB_SIZE)
                else:
                    TEXT.build_vocab(train_data,  max_size = MAX_VOCAB_SIZE)
                LABEL.build_vocab(train_data)
                print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
                print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

                train_iterator, test_iterator = data.BucketIterator.splits(
                                                            (train_data, test_data),
                                                            batch_size = args.batch_size,
                                                            sort_key=lambda x: len(x.prediction), # on what attribute the text should be sorted
                                                            sort_within_batch = True,
                                                            device = device)
                INPUT_DIM = len(TEXT.vocab)
                EMBEDDING_DIM = 100
                HIDDEN_DIM = 256
                OUTPUT_DIM = 1
                N_LAYERS = 2
                BIDIRECTIONAL = True
                DROPOUT = 0.5
                PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
                #print(PAD_IDX)

                model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

                #print(f'The model has {count_parameters(model):,} trainable parameters')

                if args.use_glove:
                    pretrained_embeddings = TEXT.vocab.vectors
                    print(pretrained_embeddings.shape)
                    model.embedding.weight.data.copy_(pretrained_embeddings)

                UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
                model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
                model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

                # Training #
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = nn.BCEWithLogitsLoss()

                model = model.to(device)
                criterion = criterion.to(device)

                N_EPOCHS = args.num_epochs

                best_valid_acc = float(0)

                train_proc = []
                valid_loss, valid_acc, avg_score, young_acc, old_acc, young_score_avg, old_score_avg = evaluate(model, test_iterator, criterion, args.batch_size, TEXT, args)
                rand_score_list.append(avg_score)

                for epoch in range(N_EPOCHS):

                    train_loss, train_acc, train_proc = train(model, train_iterator, optimizer, criterion, train_proc)

                valid_loss, valid_acc, avg_score, young_acc, old_acc, young_score, old_score = evaluate(model, test_iterator, criterion, args.batch_size, TEXT, args)
                val_acc_list.append(valid_acc)
                young_acc_list.append(young_acc)
                old_acc_list.append(old_acc)
                score_list.append(avg_score)
                young_score_list.append(young_score)
                old_score_list.append(old_score)
                #print("Average score:", avg_score)

            old_avg_acc = sum(old_acc_list) / len(old_acc_list)
            young_avg_acc = sum(young_acc_list) / len(young_acc_list)
            avg_score = sum(score_list) / len(score_list)
            young_avg_score = sum(young_score_list) / len(young_score_list)
            old_avg_score = sum(old_score_list) / len(old_score_list)

            print('########## Results ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f"\t Feyoung score: {old_avg_score*100:.2f}%")
            #print(f"\t Male score: {young_avg_score*100:.2f}%")
            #print('!Random avg score', score_list, sum(rand_score_list) / len(rand_score_list))
            print('#############################')

    ########### MODEL LIC score ###########
    if args.calc_model_leak:
        print('--- calc MODEL Leakage ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('--- task is Captioning ---')
            d_train, d_test = make_train_test_split(args, selected_cap_age_entries)

            #!!! for qualitative !!!
            flag_imid_ = 0
            if args.mask_age_words:
                with open('bias_data/train.csv', 'w') as f:
                    writer = csv.writer(f)
                    for i, entry in enumerate(d_train):
                        if entry['bb_age'] == 'Male':
                            age = 0
                        else:
                            age = 1
                        ctokens = word_tokenize(entry['pred'])
                        new_list = []
                        for t in ctokens:
                            if t in age_words:
                                new_list.append('ageword')
                            else:
                                new_list.append(t)      
                        new_sent = ' '.join([c for c in new_list])
                        if i <= 5 and args.seed == 0:
                            print(new_sent)

                        writer.writerow([new_sent.strip(), age, entry['img_id']])

                with open('bias_data/test.csv', 'w') as f:
                    writer = csv.writer(f)
                    test_imid_list = []
                    for i, entry in enumerate(d_test):
                        test_imid_list.append(entry['img_id'])
                        if entry['bb_age'] == 'Young':
                            age = 0
                        else:
                            age = 1

                        ctokens = word_tokenize(entry['pred'])
                        new_list = []
                        for t in ctokens:
                            if t in age_words:
                                new_list.append('ageword')
                            else:
                                new_list.append(t)      
                        new_sent = ' '.join([c for c in new_list])
      
                        writer.writerow([new_sent.strip(), age, entry['img_id']])

            else:
                print("!! SHOULD MASK AGE WORDS")
    

        nlp = spacy.load("en_core_web_sm")

        TEXT = data.Field(sequential=True, 
                       tokenize='spacy', 
                       tokenizer_language='en_core_web_sm',
                       include_lengths=True, 
                       use_vocab=True)
        LABEL = data.Field(sequential=False, 
                         use_vocab=False, 
                         pad_token=None, 
                         unk_token=None,
                         )
        IMID = data.Field(sequential=False,
                         use_vocab=False,
                         pad_token=None,
                         unk_token=None,
                         )



        train_val_fields = [
            ('prediction', TEXT), # process it as text
            ('label', LABEL), # process it as label
            ('imid', IMID)
        ]

        train_data, test_data = torchtext.legacy.data.TabularDataset.splits(path='bias_data/',train='train.csv', test='test.csv',
                                                                            format='csv', fields=train_val_fields)

        #ex = train_data[1]
        #print(ex.prediction, ex.label)

        MAX_VOCAB_SIZE = 25000

        if args.save_model_vocab:
            TEXT.build_vocab(train_data, test_data, max_size = MAX_VOCAB_SIZE)
            vocab_itos_list = TEXT.vocab.itos
            file_name = '/bias-vl/%s_vocab.pkl' %args.cap_model
            pickle.dump(vocab_itos_list, open(file_name, 'wb'))
            print('--- Saved vocab ---')

        if args.use_glove:
            print("-- Use GloVe")
            TEXT.build_vocab(train_data, vectors = "glove.6B.100d",  max_size = MAX_VOCAB_SIZE)
        else:
            TEXT.build_vocab(train_data,  max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
        #print(LABEL.vocab.stoi)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, test_iterator = data.BucketIterator.splits(
                                                            (train_data, test_data), 
                                                            batch_size = args.batch_size,
                                                            sort_key=lambda x: len(x.prediction), # on what attribute the text should be sorted
                                                            sort_within_batch = True,
                                                            device = device)
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        #print(PAD_IDX)

        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

        #print(f'The model has {count_parameters(model):,} trainable parameters')

        if args.use_glove:
            pretrained_embeddings = TEXT.vocab.vectors
            print(pretrained_embeddings.shape)
            model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        # Training #
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = args.num_epochs

        train_proc = []
        for epoch in range(N_EPOCHS):

            train_loss, train_acc, train_proc = train(model, train_iterator, optimizer, criterion, train_proc)

        valid_loss, valid_acc, avg_score, young_acc, old_acc, young_score, old_score  = evaluate(model, test_iterator, criterion, args.batch_size, TEXT, args)
        print('########## Results ##########')
        print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
        #print(f'\t Male. score: {young_score*100:.2f}%')
        #print(f'\t Feyoung. score: {old_score*100:.2f}%')
        print('#############################')
        print()

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("---Start---")
    print('Seed:', args.seed)
    print("Epoch:", args.num_epochs)
    print("Learning rate:", args.learning_rate)
    print("Use GLoVe:", args.use_glove)
    print("Task:", args.task)
    if args.task == 'captioning' and args.calc_model_leak:
        print("Captioning model:", args.cap_model)
    print("Protected attribute: Age")

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)
    print()

    main(args)