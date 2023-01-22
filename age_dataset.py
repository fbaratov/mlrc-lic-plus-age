'''
This script is used for building torch datasets for bert model.
As bert model is using a special tokenizer, we need this done here.
This is authors code, copied to another file and changed for the
purposes of age attribute.
'''

import argparse
import pickle
import nltk
import numpy as np
import json
import os
import pprint
#from pycocotools.coco import COCO
#import pylab
from nltk.tokenize import word_tokenize
import random
from io import open
import sys
import torch
from torch import nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange



        
class BERT_ANN_leak_data(data.Dataset):
    '''
    Bert human annotation leak data.
    '''
    def __init__(self, d_train, d_test, args, age_task_mw_entries, age_words, tokenizer, max_seq_length, split, caption_ind=None):
        '''
        Initialize the class with dataset for training and testing data for age. Human annotations
        '''
        self.task = args.task
        self.age_task_mw_entries = age_task_mw_entries
        self.cap_ind = caption_ind
        self.split = split
        self.age_words = age_words

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.d_train, self.d_test = d_train, d_test 

        # If we need to align vocab for human dataset, then use the models vocab.
        self.align_vocab = args.align_vocab
        if self.align_vocab:
            self.model_vocab = pickle.load(open('./bias_data/model_vocab/%s_vocab.pkl' %args.cap_model, 'rb'))
            print('len(self.model_vocab):', len(self.model_vocab))

    def __len__(self):
        '''
        Returns the length of iterator for train and test set.
        '''
        if self.split == 'train':
            return len(self.d_train)
        elif self.split == 'test':
            return len(self.d_test)
        else:
            raise Exception("Sorry, you must specify the split as 'train' or 'test'")

    def __getitem__(self, index):
        '''
        Indexing function for the iterator.
        Arguments
        ---------
        index : int
            The index to fetch. Similar to array indexing array[1]
        '''

        # Get the entries depending on the train or test argument.
        if self.split == 'train':
            entries = self.d_train
        elif self.split == 'test':
            entries = self.d_test
        else:
            raise Exception("Sorry, you must specify the split as 'train' or 'test'")

        entry = entries[index]
        img_id = entry['img_id']

        # Retrive the age from the entry.
        # We have three entries, young, old and unknown.
        # Do not retrive the unknown entry.
        age = entry['bb_age']
        if age == 'Young':
            age_target = torch.tensor(0)
        elif age == 'Old':
            age_target = torch.tensor(1)
          

        if self.task == 'captioning':
            ctokens = word_tokenize(entry['caption_list'][self.cap_ind].lower())
            new_list = []
            for t in ctokens:
                if t in self.age_words:
                    new_list.append('[MASK]')
                elif self.align_vocab:
                    if t not in self.model_vocab:
                        new_list.append('[UNK]')
                    else:
                        new_list.append(t)
                else:
                    new_list.append(t)
            new_sent = ' '.join([c for c in new_list])

            encoded_dict = self.tokenizer.encode_plus(new_sent, add_special_tokens=True, truncation=True, max_length=self.max_seq_length, 
                                                    padding='max_length', return_attention_mask=True, return_tensors='pt')


        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        token_type_ids = token_type_ids.view(self.max_seq_length)

        input_ids = input_ids.view(self.max_seq_length)
        attention_mask = attention_mask.view(self.max_seq_length)

        return input_ids, attention_mask, token_type_ids, age_target, img_id



class BERT_MODEL_leak_data(data.Dataset):
    '''
    Bert human annotation leak data.
    '''
    def __init__(self, d_train, d_test, args, age_task_mw_entries, age_words, tokenizer, max_seq_length, split, caption_ind=None):
        '''
        Initialize the class with dataset for training and testing data for age. Human annotations
        '''
        self.task = args.task
        self.age_task_mw_entries = age_task_mw_entries
        self.cap_ind = caption_ind
        self.split = split
        self.age_words = age_words

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.d_train, self.d_test = d_train, d_test 

    def __len__(self):
        '''
        Returns the length of iterator for train and test set.
        '''
        if self.split == 'train':
            return len(self.d_train)
        elif self.split == 'test':
            return len(self.d_test)
        else:
            raise Exception("Sorry, you must specify the split as 'train' or 'test'")
          
    def __getitem__(self, index):
        # Get the entries depending on the train or test argument.
        if self.split == 'train':
            entries = self.d_train
        elif self.split == 'test':
            entries = self.d_test
        else:
            raise Exception("Sorry, you must specify the split as 'train' or 'test'")
        
        entry = entries[index]
        img_id = entry['img_id']

        # Get the age retrive the age target.
        age = entry['bb_age']
        if age == 'Young':
            age_target = torch.tensor(0)
        elif age == 'Old':
            age_target = torch.tensor(1)

        if self.task == 'captioning':
            c_pred_tokens = word_tokenize(entry['pred'].lower())
            new_list = []
            for t in c_pred_tokens:
                if t in self.age_words:
                    new_list.append('[MASK]')
                else:
                    new_list.append(t)
            new_sent = ' '.join([c for c in new_list])

            encoded_dict = self.tokenizer.encode_plus(new_sent, add_special_tokens=True, truncation=True, max_length=self.max_seq_length,
                                                    padding='max_length', return_attention_mask=True, return_tensors='pt')


        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        token_type_ids = encoded_dict['token_type_ids']
        token_type_ids = token_type_ids.view(self.max_seq_length)

        input_ids = input_ids.view(self.max_seq_length)
        attention_mask = attention_mask.view(self.max_seq_length)

        return input_ids, attention_mask, token_type_ids, age_target, img_id