#!/usr/local/Cellar/python/3.7.4_1
# -*- coding: utf-8 -*-
# @File    : metric.py
# @Author  : 姜小帅
# @Moto    : 良好的阶段性收获是坚持的重要动力之一
# @Contract: Mason_Jay@163.com
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def flat_accuracy(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def flat_f1(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()

    return f1_score(labels_flat, pred_flat)

