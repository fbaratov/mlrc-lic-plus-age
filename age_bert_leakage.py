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

from string import punctuation

import pickle
import pandas as pd
import random
from collections import namedtuple

# import age variables from utils
from age_utils import (
  young_words,
  old_words,
  age_words
)
# import age functiond from utils
from age_utils import (
  gender_pickle_generator,
  race_pickle_generator,
  label_human_caption,
  label_human_annotations,
  match_labels,
  make_train_test_split,
  save_leak_model
)
from age_dataset import BERT_ANN_leak_data, BERT_MODEL_leak_data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str)
    parser.add_argument("--gender_or_race", default='gender', type=str)
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--calc_mw_acc", default=True, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_age_words", default=True, type=bool)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--store_topk_age_pred", default=False, type=bool)
    parser.add_argument("--topk_age_pred", default=50, type=int)
    parser.add_argument("--calc_score", default=True, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--optimizer", default='adamw', type=str, help="adamw or adam")
    parser.add_argument("--adam_correct_bias", default=True, type=bool)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.98, type=float, help="0.999:huggingface, 0.98:RoBERTa paper")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer. 1e-8:first, 1e-6:RoBERTa paper")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight deay if we apply some. 0.001:first, 0.01:RoBERTa")
    parser.add_argument("--coco_lk_model_dir", default='/Bias/leakage/', type=str)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)

    # Add an argument to save the model after training is finished.
    parser.add_argument('--save_model', default = True, type = bool)
    parser.add_argument('--every', default = 5, type = int, help = 'Saving period of the model in epochs')

    return parser

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def calc_random_acc_score(args, model, test_dataloader):
    print("--- Random guess --")
    model = model.cuda()
    optimizer = None
    epoch = None
    val_loss, val_acc, val_young_acc, val_old_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)

    return val_acc, val_loss, val_young_acc, val_old_acc, avg_score



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
        if args.save_model:
              if epoch % args.every == 0:
                path = "saved_models/{}"
                if args.freeze_bert:
                  file_name = "age_annotation_{}_model_pretrained_bert_{}_seed_{}_epoch_{}.pt"
                else:
                  file_name = "age_annotation_{}_model_bert_{}_seed_{}_epoch_{}.pt"
                save_leak_model(model,epoch, file_name,path, args)
                 

    print("Finish training")
    print('{0}: train acc: {1:2f}'.format(epoch, train_acc))
    # We also save the model after training is finished
    if args.freeze_bert:
      file_name = "age_annotation_{}_model_pretrained_bert_{}_seed_{}_epoch_{}.pt"
    else:
      file_name = "age_annotation_{}_model_bert_{}_seed_{}_epoch_{}.pt"
    path = "saved_models/{}"
    save_leak_model(model,epoch, file_name,path, args)
    # validation
    val_loss, val_acc, val_young_acc, val_old_acc, avg_score = calc_leak_epoch_pass(epoch, test_dataloader, model, optimizer, False, print_every=500)
    print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))
    if args.calc_mw_acc:
        print('val, {0}, val loss: {1:.2f}, Young val acc: {2:.2f}'.format(epoch, val_loss*100, val_young_acc *100))
        print('val, {0}, val loss: {1:.2f}, Old val acc: {2:.2f}'.format(epoch, val_loss*100, val_old_acc *100))

    return val_acc, val_loss, val_young_acc, val_old_acc, avg_score


def calc_leak_epoch_pass(epoch, data_loader, model, optimizer, training, print_every):
    t_loss = 0.0
    n_processed = 0
    preds = list()
    truth = list()
    young_preds_all, old_preds_all = list(), list()
    young_truth_all, old_truth_all = list(), list()

    if training:
        model.train()
    else:
        model.eval()

    if args.store_topk_age_pred:
        all_young_pred_values, all_old_pred_values = [], []
        all_young_inputs, all_old_inputs = [], []

    total_score = 0 # for calculate scores

    cnt_data = 0
    for ind, (input_ids, attention_mask, token_type_ids, age_target, img_id) in enumerate(data_loader): # images are not provided
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

        age_target = torch.squeeze(age_target).cuda()
        predictions = model(input_ids, attention_mask, token_type_ids)
        cnt_data += predictions.size(0)

        loss = F.cross_entropy(predictions, age_target, reduction='mean')

        if not training and args.store_topk_age_pred:
            pred_values = np.amax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1).tolist()
            pred_ages = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)

            for pv, pg, imid, ids in zip(pred_values, pred_ages, img_id, input_ids):
                tokens = model.tokenizer.convert_ids_to_tokens(ids)
                text = model.tokenizer.convert_tokens_to_string(tokens)
                text = text.replace('[PAD]', '')
                if pg == 0:
                    all_young_pred_values.append(pv)
                    all_young_inputs.append({'img_id': imid, 'text': text})
                else:
                    all_old_pred_values.append(pv)
                    all_old_inputs.append({'img_id': imid, 'text': text})

        if not training and args.calc_score:
            pred_ages = np.argmax(F.softmax(predictions, dim=1).cpu().detach(), axis=1)
            age_target = age_target.cpu().detach()
            correct = torch.eq(pred_ages, age_target)
            #if ind == 0:
            #    print('correct:', correct, correct.shape)

            pred_score_tensor = torch.zeros_like(correct, dtype=float)
            for i in range(pred_score_tensor.size(0)):
                young_score = F.softmax(predictions, dim=1).cpu().detach()[i,0]
                old_score = F.softmax(predictions, dim=1).cpu().detach()[i,1]
                if young_score >= old_score:
                    pred_score = young_score
                else:
                    pred_score = old_score

                pred_score_tensor[i] = pred_score

            scores_tensor = correct.int() * pred_score_tensor
            correct_score_sum = torch.sum(scores_tensor)
            total_score += correct_score_sum.item()

        predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
        preds += predictions.tolist()
        truth += age_target.cpu().numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(age_target)

        if (ind + 1) % print_every == 0 and training:
            print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

        if args.calc_mw_acc and not training:
            young_target_ind = [i for i, x in enumerate(age_target.cpu().numpy().tolist()) if x == 0]
            old_target_ind = [i for i, x in enumerate(age_target.cpu().numpy().tolist()) if x == 1]
            young_pred = [*itemgetter(*young_target_ind)(predictions.tolist())]
            old_pred = [*itemgetter(*old_target_ind)(predictions.tolist())]
            young_target = [*itemgetter(*young_target_ind)(age_target.cpu().numpy().tolist())]
            old_target = [*itemgetter(*old_target_ind)(age_target.cpu().numpy().tolist())]
            young_preds_all += young_pred
            young_truth_all += young_target
            old_preds_all += old_pred
            old_truth_all += old_target

    acc = accuracy_score(truth, preds)

    if args.calc_mw_acc and not training:
        young_acc = accuracy_score(young_truth_all, young_preds_all)
        old_acc = accuracy_score(old_truth_all, old_preds_all)
    else:
        young_acc, old_acc = None, None

    return t_loss / n_processed, acc, young_acc, old_acc, total_score / cnt_data



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
            young_acc_list, old_acc_list = [], []
            rand_acc_list = []
            rand_score_list = []
            for caption_ind in range(5):
                trainANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, age_val_obj_cap_entries, age_words, tokenizer,
                                                args.max_seq_length, split='train', caption_ind=caption_ind)
                testANNCAPobject = BERT_ANN_leak_data(d_train, d_test, args, age_val_obj_cap_entries, age_words, tokenizer,
                                                args.max_seq_length, split='test', caption_ind=caption_ind)
                train_dataloader = torch.utils.data.DataLoader(trainANNCAPobject, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
                test_dataloader = torch.utils.data.DataLoader(testANNCAPobject, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
                # initialize gender classifier (works for age too)
                model = BERT_GenderClassifier(args, tokenizer)
                # calculate random predictions
                val_acc, val_loss, val_young_acc, val_old_acc, avg_score = calc_random_acc_score(args, model, test_dataloader)
                rand_acc_list.append(val_acc)
                rand_score_list.append(avg_score)
                # train and test
                val_acc, val_loss, val_young_acc, val_old_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)
                val_acc_list.append(val_acc)
                young_acc_list.append(val_young_acc)
                old_acc_list.append(val_old_acc)
                score_list.append(avg_score)

            old_avg_acc = sum(old_acc_list) / len(old_acc_list)
            young_avg_acc = sum(young_acc_list) / len(young_acc_list)
            avg_score = sum(score_list) / len(score_list)
            print('########### Reluts ##########')
            print(f"LIC score (LIC_D): {avg_score*100:.2f}%")
            #print(f"\t Old Accuracy: {old_avg_acc*100:.2f}%")
            #print(f"\t Young Accuracy: {young_avg_acc*100:.2f}%")
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
            rand_val_acc, rand_val_loss, rand_val_young_acc, rand_val_old_acc, rand_avg_score = calc_random_acc_score(args, model, test_dataloader)
            # train and test
            val_acc, val_loss, val_young_acc, val_old_acc, avg_score = calc_leak(args, model, train_dataloader, test_dataloader)

            print('########### Reluts ##########')
            print(f'LIC score (LIC_M): {avg_score*100:.2f}%')
            #print(f'\t Young. Acc: {val_young_acc*100:.2f}%')
            #print(f'\t Old. Acc: {val_old_acc*100:.2f}%')
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
    print("Protected attribute: Age")
    print("Save Model : ", args.save_model)
    if args.save_model:
      print("Saving Every : ", args.every)
    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)

    print()

    main(args)


