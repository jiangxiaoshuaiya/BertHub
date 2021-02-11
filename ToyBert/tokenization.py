#!/usr/local/Cellar/python/3.7.4_1
# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 12:30 上午
# @File    : tokenization.py
# @Author  : 姜小帅
# @Moto    : 良好的阶段性收获是坚持的重要动力之一
# @Contract: Mason_Jay@163.com
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, random_split, TensorDataset


def process(context, name, batch_size, max_length, threshold=None):
    '''
    Parameters@
      context: list, a matrix which dimensional equals to two
      tokenizer: tokenizer, as its name, use to encode sentence
      batch_size: int, batch size of one train
      max_length: int, max length of sequence
      threshold: float, use to split train data, range from 0 to 1

    Returns@
      dataloader of train set and valid set while need to split data, otherwise, test set
    '''

    tokenizer = BertTokenizer.from_pretrained(name)
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for text in context:
        encode_dic = tokenizer.encode_plus(
            text=text[1],
            text_pair=text[3],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            pad_to_multiple_of=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        input_ids.append(encode_dic['input_ids'])
        attention_masks.append(encode_dic['attention_mask'])
        token_type_ids.append(encode_dic['token_type_ids'])
        labels.append([int(text[4])] if text[4] is not None else [-1])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels = torch.tensor(labels)

    # pack up and transform to form of dataloader, which is feeded to model,
    # return train set and vaild set while  threshold is not None,
    # otherwise, predict dataloader
    dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)

    def _init_fn(worker_id):
        np.random.seed(int(2020))
    if threshold:

        if threshold > 0 or threshold < 1:
            train_size = int(threshold * len(dataset))
            val_size = len(dataset) - train_size
            train_data, val_data = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_data, batch_size=batch_size,
                                      sampler=RandomSampler(train_data), num_workers=2, worker_init_fn=_init_fn)
            val_loader = DataLoader(val_data, batch_size=batch_size,
                                    sampler=SequentialSampler(val_data), num_workers=2, worker_init_fn=_init_fn)

            return train_loader, val_loader

        else:
            print('NumericalError: threshold out of range, a correct value must between (0,1)')
    else:

        predict_loader = DataLoader(dataset, batch_size=batch_size, 
                                    sampler=SequentialSampler(dataset), num_workers=2, worker_init_fn=_init_fn)

        return predict_loader

