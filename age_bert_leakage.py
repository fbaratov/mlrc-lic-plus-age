import torch
import csv
import spacy
import re
import pickle
import random
import csv
from nltk import word_tokenize
import nltk
nltk.download('punkt')
import time

import argparse
import os
import pprint
import numpy as np
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
import pickle
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import transformers as tf
from transformers import BertTokenizer
from transformers import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, get_linear_schedule_with_warmup

import torch.utils.data as data
from transformers import BertModel
from transformers import BertPreTrainedModel

from model import BERT_GenderClassifier
from bias_dataset import BERT_ANN_leak_data, BERT_MODEL_leak_data

from string import punctuation

import pickle
import pandas as pd
import random
from collections import namedtuple

# import unchanged functions from original file, just to be clear what is modified and what isn't
from gender_bert_leakage import binary_accuracy

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


def calc_random_acc_score(args, model, test_dataloader):
    print("--- Random guess --")
    model = model.cuda()
    optimizer = None
    epoch = None
    val_loss, val_acc, val_male_acc, val_female_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)

    return val_acc, val_loss, val_male_acc, val_female_acc, avg_score



def calc_leak(args, model, train_dataloader, test_dataloader):
    model = model.cuda()
    print("Num of Trainable Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-5)
    elif args.optimizer == 'adamw':
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            #{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), correct_bias=args.adam_correct_bias, eps=args.adam_epsilon)

    train_loss_arr = list()
    train_acc_arr = list()

    # training
    for epoch in range(args.num_epochs):
        # train
        train_loss, train_acc, _, _, _ = calc_leak_epoch_pass(epoch, train_dataloader, model, optimizer, True, print_every=500)
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        if epoch % 5 == 0:
            print('train, {0}, train loss: {1:.2f}, train acc: {2:.2f}'.format(epoch, \
                train_loss*100, train_acc*100))

    print("Finish training")
    print('{0}: train acc: {1:2f}'.format(epoch, train_acc))

    # validation
    val_loss, val_acc, val_male_acc, val_female_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)
    print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))
    if args.calc_mw_acc:
        print('val, {0}, val loss: {1:.2f}, Male val acc: {2:.2f}'.format(epoch, val_loss*100, val_male_acc *100))
        print('val, {0}, val loss: {1:.2f}, Feale val acc: {2:.2f}'.format(epoch, val_loss*100, val_female_acc *100))

    return val_acc, val_loss, val_male_acc, val_female_acc, avg_score


def calc_leak_epoch_pass(epoch, data_loader, model, optimizer, training, print_every):
    t_loss = 0.0
    n_processed = 0
    preds = list()
    truth = list()
    male_preds_all, female_preds_all = list(), list()
    male_truth_all, female_truth_all = list(), list()

    if training:
        model.train()
    else:
        model.eval()

    if args.store_topk_gender_pred:
        all_male_pred_values, all_female_pred_values = [], []
        all_male_inputs, all_female_inputs = [], []

    total_score = 0 # for calculate scores

    cnt_data = 0
    for ind, (input_ids, attention_mask, token_type_ids, gender_target, img_id) in enumerate(data_loader): # images are not provided
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

        gender_target = torch.squeeze(gender_target).cuda()
        predictions = model(input_ids, attention_mask, token_type_ids)
        cnt_data += predictions.size(0)

        loss = F.cross_entropy(predictions, gender_target, reduction='mean')

        if not training and args.store_topk_gender_pred:
            pred_values = np.amax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
            pred_genders = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

            for pv, pg, imid, ids in zip(pred_values, pred_genders, img_id, input_ids):
                tokens = model.tokenizer.convert_ids_to_tokens(ids)
                text = model.tokenizer.convert_tokens_to_string(tokens)
                text = text.replace('[PAD]', '')
                if pg == 0:
                    all_male_pred_values.append(pv)
                    all_male_inputs.append({'img_id': imid, 'text': text})
                else:
                    all_female_pred_values.append(pv)
                    all_female_inputs.append({'img_id': imid, 'text': text})

        if not training and args.calc_score:
            pred_genders = np.argmax(F.softmax(predictions, dim=1).cpu().detach(), axis=1)
            gender_target = gender_target.cpu().detach()
            correct = torch.eq(pred_genders, gender_target)
            #if ind == 0:
            #    print('correct:', correct, correct.shape)

            pred_score_tensor = torch.zeros_like(correct, dtype=float)
            for i in range(pred_score_tensor.size(0)):
                male_score = F.softmax(predictions, dim=1).cpu().detach()[i,0]
                female_score = F.softmax(predictions, dim=1).cpu().detach()[i,1]
                if male_score >= female_score:
                    pred_score = male_score
                else:
                    pred_score = female_score

                pred_score_tensor[i] = pred_score

            scores_tensor = correct.int() * pred_score_tensor
            correct_score_sum = torch.sum(scores_tensor)
            total_score += correct_score_sum.item()

        predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
        preds += predictions.tolist()
        truth += gender_target.cpu().numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(gender_target)

        if (ind + 1) % print_every == 0 and training:
            print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

        if args.calc_mw_acc and not training:
            male_target_ind = [i for i, x in enumerate(gender_target.cpu().numpy().tolist()) if x == 0]
            female_target_ind = [i for i, x in enumerate(gender_target.cpu().numpy().tolist()) if x == 1]
            male_pred = [*itemgetter(*male_target_ind)(predictions.tolist())]
            female_pred = [*itemgetter(*female_target_ind)(predictions.tolist())]
            male_target = [*itemgetter(*male_target_ind)(gender_target.cpu().numpy().tolist())]
            female_target = [*itemgetter(*female_target_ind)(gender_target.cpu().numpy().tolist())]
            male_preds_all += male_pred
            male_truth_all += male_target
            female_preds_all += female_pred
            female_truth_all += female_target

    acc = accuracy_score(truth, preds)

    if args.calc_mw_acc and not training:
        male_acc = accuracy_score(male_truth_all, male_preds_all)
        female_acc = accuracy_score(female_truth_all, female_preds_all)
    else:
        male_acc, female_acc = None, None

    return t_loss / n_processed, acc, male_acc, female_acc, total_score / cnt_data



def main(args):
    '''
    Modified version of main from gender_bert_leakage.

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN LIC score ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split(args, age_val_obj_cap_entries)
            val_acc_list = []
            score_list = []
            male_acc_list, female_acc_list = [], []
            rand_acc_list = []
            rand_score_list = []
            for caption_ind in range(5):
                trainANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, age_val_obj_cap_entries, age_words, tokenizer,
                                                args.max_seq_length, split='train', caption_ind=caption_ind)
                testANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, age_val_obj_cap_entries, age_words, tokenizer,
                                                args.max_seq_length, split='test', caption_ind=caption_ind)
                train_dataloader = torch.utils.data.DataLoader(trainANNCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
                test_dataloader = torch.utils.data.DataLoader(testANNCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
                # initialize gender classifier
                model = BERT_GenderClassifier(args, tokenizer)
                # calculate random predictions
                val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_random_acc_score(args, model, test_dataloader)
                rand_acc_list.append(val_acc)
                rand_score_list.append(avg_score)
                # train and test
                val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)
                val_acc_list.append(val_acc)
                male_acc_list.append(val_male_acc)
                female_acc_list.append(val_female_acc)
                score_list.append(avg_score)

            female_avg_acc = sum(female_acc_list) / len(female_acc_list)
            male_avg_acc = sum(male_acc_list) / len(male_acc_list)
            avg_score = sum(score_list) / len(score_list)
            print('########### Reluts ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f"\t Female Accuracy: {female_avg_acc*100:.2f}%")
            #print(f"\t Male Accuracy: {male_avg_acc*100:.2f}%")
            print('#############################')



    ##################### MODEL LIC score #######################
    if args.calc_model_leak:
        print('--- calc MODEL LIC score---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- Task is Captioning --')
            d_train, d_test = make_train_test_split(args, selected_cap_age_entries)
            trainMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_age_entries, age_words, tokenizer,
                                                args.max_seq_length, split='train')
            testMODELCAPobject = BERT_MODEL_leak_data(d_train, d_test, args, selected_cap_age_entries, age_words, tokenizer,
                                                args.max_seq_length, split='test')
            train_dataloader = torch.utils.data.DataLoader(trainMODELCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
            test_dataloader = torch.utils.data.DataLoader(testMODELCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
            # initialize gender classifier
            model = BERT_GenderClassifier(args, tokenizer)
            # calculate random predictions
            rand_val_acc, rand_val_loss, rand_val_male_acc, rand_val_female_acc, rand_avg_score = calc_random_acc_score(args, model, test_dataloader)
            # train and test
            val_acc, val_loss, val_male_acc, val_female_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)

            print('########### Reluts ##########')
            print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
            #print(f'\t Male. Acc: {val_male_acc*100:.2f}%')
            #print(f'\t Female. Acc: {val_female_acc*100:.2f}%')
            print('#############################')




if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print()
    print("---Start---")
    print('Seed:', args.seed)
    print("Epoch:", args.num_epochs)
    print("Freeze BERT:", args.freeze_bert)
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)
    print("Calculate score:", args.calc_score)
    print("Task:", args.task)
    if args.task == 'captioning' and args.calc_model_leak:
        print("Captioning model:", args.cap_model)
    print("Gender or Race:", args.gender_or_race)

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)

    print()

    main(args)

