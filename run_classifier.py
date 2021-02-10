#!/usr/local/Cellar/python/3.7.4_1
# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 12:17 上午
# @File    : run_classifier.py
# @Author  : 姜小帅
# @Moto    : 良好的阶段性收获是坚持的重要动力之一
# @Contract: Mason_Jay@163.com
import os
import torch
import random
import numpy as np
from datetime import time

from ToyBert.utils import load_data, format_time
from ToyBert.metric import flat_f1
from ToyBert.tokenization import process
from ToyBert.model import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer


def main():
    '''
    NOTES:
    1、This is the main function for training a model to achieve your downstream 
    task in natural language processing, such as question&answer match, sequence 
    classification and so on.
    2、You could load any other pretrained model which huggingface have supported, 
    for example: hfl/chinese-bert-wwm.
    3、Happy for sharing this project to others, if you also do, light the star up and
    bring a link. 
    4、Great wishes in modeling, enjoy it !!!
    '''
    PATH = 'drive/MyDrive/drive'
    SEED = 2020
    EPOCHS = 5
    BATCH_SIZE = 32
    MAX_LENGTH = 256
    LEARNING_RATE = 1e-5
    NAME = 'hfl/chinese-bert-wwm'

    fix_seed(SEED)
    train = load_data(PATH, train_test='train')
    test = load_data(PATH, train_test='test')
    print('train example', train[0:5])
    print('test example', test[0:5])

    train_dataloader, valid_dataloader = process(train, NAME, BATCH_SIZE, MAX_LENGTH, threshold=0.8)
    test_dataloader = process(test, NAME, BATCH_SIZE, MAX_LENGTH)
    print('data processing is done !!')

    bert = BertForSequenceClassification(NAME)
    optimizer = AdamW(bert.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    # change learning rate dynamically in total steps,
    # during warmup phase and train period
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    bert.cuda()

    for epoch in range(EPOCHS):

        print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
        print('Training...')
        bert.train()
        start_train = time.time()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):

            if step % 50 == 0:
                elapsed = format_time(time.time() - start_train)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            batch_input_ids = batch[0].cuda()
            batch_token_type_ids = batch[1].cuda()
            batch_attention_masks = batch[2].cuda()
            batch_labels = batch[3].cuda()

            logits, loss = bert(batch_input_ids,
                                batch_attention_masks, batch_token_type_ids, labels=batch_labels)
            bert.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - start_train)
        print("  Average training CrossEntropyLoss: {0:.2f}".format(average_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print('Running Validation...')
        bert.eval()
        start_eval = time.time()
        total_eval_loss = 0
        total_eval_f1 = 0
        for step, batch in enumerate(valid_dataloader):

            if step % 50 == 0:
                elapsed = format_time(time.time() - start_train)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(valid_dataloader), elapsed))

            batch_input_ids = batch[0].cuda()
            batch_token_type_ids = batch[1].cuda()
            batch_attention_masks = batch[2].cuda()
            batch_labels = batch[3].cuda()

            with torch.no_grad():
                logits, loss = bert(batch_input_ids,
                                    batch_attention_masks, batch_token_type_ids, labels=batch_labels)
                total_eval_loss += loss.item()

        average_eval_loss = total_eval_loss / len(valid_dataloader)
        total_eval_f1 += flat_f1(logits, batch_labels)

        validation_time = format_time(time.time() - start_eval)
        print("  Average eval CrossEntropyLoss: {0:.2f}".format(average_eval_loss))
        print("  Eval f1 score: {0:.2f}".format(total_eval_f1))
        print('  Validation took: {:}'.format(validation_time))


if __name__ == '__main__':
    main()

