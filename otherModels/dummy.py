#!/usr/local/env python
"""
Scorer for the Fake News Challenge
 - @bgalbraith

Submission is a CSV with the following fields: Headline, Body ID, Stance
where Stance is in {agree, disagree, discuss, unrelated}

Scoring is as follows:
  +0.25 for each correct unrelated
  +0.25 for each correct related (label is any of agree, disagree, discuss)
  +0.75 for each correct agree, disagree, discuss
"""
from __future__ import division
import csv
import sys
import os

import numpy as np
import torch


FIELDNAMES = ['Headline', 'Body ID', 'Stance']
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]

FIELDNAMESBODIES = ['Body ID' , 'articleBody']

USAGE = """
FakeNewsChallenge FNC-1 scorer - version 1.0
Usage: python scorer.py gold_labels test_labels

  gold_labels - CSV file with reference GOLD stance labels
  test_labels - CSV file with predicted stance labels

The scorer will provide three scores: MAX, NULL, and TEST
  MAX  - the best possible score (100% accuracy)
  NULL - score as if all predicted stances were unrelated
  TEST - score based on the provided predictions
"""

ERROR_MISMATCH = """
ERROR: Entry mismatch at line {}
 [expected] Headline: {} // Body ID: {}
 [got] Headline: {} // Body ID: {}
"""

SCORE_REPORT = """
MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||\n||{:^11}||{:^11}||{:^11}||
"""


class FNCException(Exception):
    pass


def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        if g['Headline'] != t['Headline'] or g['Body ID'] != t['Body ID']:
            error = ERROR_MISMATCH.format(i+2,
                                          g['Headline'], g['Body ID'],
                                          t['Headline'], t['Body ID'])
            raise FNCException(error)
        else:
            g_stance, t_stance = g['Stance'], t['Stance']
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 'unrelated':
                    score += 0.50
            if g_stance in RELATED and t_stance in RELATED:
                score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def score_defaults(gold_labels):
    """
    Compute the "all false" baseline (all labels as unrelated) and the max
    possible score
    :param gold_labels: list containing the true labels
    :return: (null_score, best_score)
    """
    unrelated = [g for g in gold_labels if g['Stance'] == 'unrelated']
    null_score = 0.25 * len(unrelated)
    max_score = null_score + (len(gold_labels) - len(unrelated))
    return null_score, max_score


def load_dataset(filename):
    data = None
    try:
        with open(filename) as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != FIELDNAMES:
                error = 'ERROR: Incorrect headers in: {}'.format(filename)
                raise FNCException(error)
            else:
                data = list(reader)

            if data is None:
                error = 'ERROR: No data found in: {}'.format(filename)
                raise FNCException(error)
    except FileNotFoundError:
        error = "ERROR: Could not find file: {}".format(filename)
        raise FNCException(error)

    return data


def load_dataset_bodies(filename):
    data = None
    try:
        with open(filename) as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != FIELDNAMESBODIES:
                error = 'ERROR: Incorrect headers in: {}'.format(filename)
                raise FNCException(error)
            else:
                data = list(reader)

            if data is None:
                error = 'ERROR: No data found in: {}'.format(filename)
                raise FNCException(error)
    except FileNotFoundError:
        error = "ERROR: Could not find file: {}".format(filename)
        raise FNCException(error)

    return data



def print_confusion_matrix(cm):
    lines = ['CONFUSION MATRIX:']
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    lines.append("ACCURACY: {:.3f}".format(hit / total))
    print('\n'.join(lines))



def getFNC(data_path):
    s1 = {}
    s2 = {}
    target = {}

    competition_test_stances = load_dataset(os.path.join(data_path,'competition_test_stances.csv'))
    competition_test_bodies = load_dataset_bodies(os.path.join(data_path,'competition_test_bodies.csv'))

    train_stances = load_dataset(os.path.join(data_path,'train_stances.csv'))
    train_bodies = load_dataset_bodies(os.path.join(data_path,'train_bodies.csv'))

    dico_label = {'unrelated': 1,  'agree': 0, 'disagree': 0, 'discuss': 0}

    sen1 = {}
    sen2 = {}
    tar = {}
    sen1['test'] = []
    sen2['test'] = []
    tar['test'] = []
    
    sen1['train'] = []
    sen2['train'] = []
    tar['train'] = []
    
    sen1['dev'] = []
    sen2['dev'] = []
    tar['dev'] = []

    for a in competition_test_bodies:
        for b in competition_test_stances:
            if int(a['Body ID'].rstrip('\n')) == int(b['Body ID'].rstrip('\n')):
                #print("yeah")
                sen1['test'].append(b['Headline'].rstrip())
                aa = a['articleBody'].split(".")
                #sen2['test'].append(aa[0].rstrip() + " .")
                #print(aa[0].rstrip()+" .")
                #sen2['test'].append(a['articleBody'].rstrip())
                inputSentence = ""
                for i in range(0,len(aa)):
                    if i%5 == 0:                     
                        inputSentence = inputSentence + aa[i].rstrip() + " ."



                inputSentence = inputSentence + aa[len(aa) - 1].rstrip() + " ."


                sen2['test'].append(inputSentence)

                tar['test'].append(b['Stance'].rstrip()) 
                break

    kappa = 1
    for a in train_bodies:

        for b in train_stances:

            if kappa%10 == 0:
                if int(a['Body ID'].rstrip('\n')) == int(b['Body ID'].rstrip('\n')):
                    #print("yeah")
                    sen1['dev'].append(b['Headline'].rstrip())
                    #sen2['dev'].append(a['articleBody'])
                    aa = a['articleBody'].split(".")
                    inputSentence = ""
                    for i in range(0,len(aa)):
                        if i%5 == 0:                     
                            inputSentence = inputSentence + aa[i].rstrip() + " ."

                    inputSentence = inputSentence + aa[len(aa) - 1].rstrip() + " ."

                    #sen2['dev'].append(' '.join(a['articleBody'].split()))
                    sen2['dev'].append(inputSentence)
                    #print(aa[0].rstrip()+" .")
                    tar['dev'].append(b['Stance'].rstrip()) 
                    break
            else:
                if int(a['Body ID'].rstrip('\n')) == int(b['Body ID'].rstrip('\n')):
                    #print("yeah")
                    sen1['train'].append(b['Headline'].rstrip())
                    #sen2['train'].append(a['articleBody'])
                    aa = a['articleBody'].split(".")
                    #sen2['train'].append(' '.join(a['articleBody'].split()))

                    inputSentence = ""
                    for i in range(0,len(aa)):
                        if i%5 == 0:                     
                            inputSentence = inputSentence + aa[i].rstrip() + " ."


                    inputSentence = inputSentence + aa[len(aa) - 1].rstrip() + " ."

                    sen2['train'].append(inputSentence)
                    #print(aa[0].rstrip()+" .")
                    tar['train'].append(b['Stance'].rstrip()) 
                    break

        kappa = kappa + 1


    for data_type in ['train', 'dev', 'test']:

        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 sen1[data_type]]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 sen2[data_type]]

        target[data_type]['data'] = np.array([dico_label[line] for line in tar[data_type]])

        unrel = np.where(target[data_type]['data'] == 0)[0].shape
        relt = np.where(target[data_type]['data'] == 1)[0].shape
        
        print(unrel)
        print(relt)
        
        unrel = np.where(target[data_type]['data'] == 2)[0].shape
        relt = np.where(target[data_type]['data'] == 3)[0].shape
        print(unrel)
        print(relt)


        #print(target[data_type]['data'])
        #print(len(target[data_type]['data']))
        #print(len(s1[data_type]['sent']))
        #print(len(s2[data_type]['sent']))

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))




    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test
