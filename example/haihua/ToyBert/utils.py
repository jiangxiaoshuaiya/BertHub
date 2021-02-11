#!/usr/local/Cellar/python/3.7.4_1
# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : 姜小帅
# @Moto    : 良好的阶段性收获是坚持的重要动力之一
# @Contract: Mason_Jay@163.com
import os
import json
import torch
import random
import numpy as np
import datetime


# input examples for multiple choice reading comprehension
class InputExample(object):
    '''
    @Parameters
    Id: question id, a paraph maybe contains tow questions,
        one question corresponds to an instance of corresponding
    context: related paragraphs for answering questions
    pair: consist of a question and an answer, 
        form likes: question + ' ' + answer(just one choice)
    label: correct answer for question
    '''
    def __init__(self, Id, context, pair, label=-1):
        self.Id = Id
        self.context = context
        self.pair = pair
        self.label = label

# input feature for multiple choice reading comprehension
# tokenization make InputExample become InputFeature 
class InputFeature(object):
    '''
    @Parameters
    example_id: question id, a paraph maybe contains tow questions,
        one question corresponds to an instance of corresponding
    choices_features: input ids, attention_mask and token_type_ids of each example
    label: correct answer for question
    '''
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for input_ids, attention_mask, token_type_ids in choices_features
        ]
        self.label = label

def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


# a defined class of model output
class ModelOutput(object):
  def __init__(self, loss, logits, hidden_states=None, attentions=None):
      self.loss = loss
      self.logits = logits
      self.hidden_states = hidden_states
      self.attentions = attentions



def load_data(path, train_test):
    
    with open(path + '{}.json'.format(train_test)) as json_file:
        data = json.load(json_file)

    option = ['A', 'B', 'C', 'D']
    label_map = {label: idx for idx, label in enumerate(option)}

    examples = []
    for item in data:
        question = None
        answer = None
        choice = None
        Id = item['ID']
        context = item['Content']
        for qa in item['Questions']:
            Id = qa['Q_id']
            answer = label_map[qa['Answer']] if len(qa)==4 is not None else -1
            question = qa['Question']
            choice = qa['Choices']
    
            examples.append(
                InputExample(
                    Id=int(Id),
                    context=[context for i in range(len(choice))],
                    pair=[question + ' ' + i[2:] for i in choice],
                    label=answer
                )
            )
    
    return examples


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fix_seed(seed):
    '''
    This funcation help you to fix seed in train, 
    that means you could get same results in sevral times
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

