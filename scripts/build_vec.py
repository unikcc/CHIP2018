#!/usr/bin/env python
import os

import json
from gensim import models

def get_config():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess an NLI dataset')
    parser.add_argument('--config',
                        default="../config/build_vec.json",
                        help='Path to a configuration file for preprocessing')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as cfg_file:
        config = json.load(cfg_file)
    return config


def prepare_question(config):
    question_file = os.path.join(config['data_dir'], 'question_id.csv')
    doc = open(question_file, 'r').read().splitlines()
    d = {w.split(',')[0] : w.split(',')[-1] for w in doc}
    return d


def prepare_id(config):
    train_file = os.path.join(config['data_dir'], 'train.csv')
    doc = open(train_file, 'r').read().splitlines()[1:]
    doc =[w[:-2] for w in doc]
    train_ids = (','.join(doc)).split(',')
    return train_ids


def prepare_data(d, ids):
    # list  for sentence
    data = [d[w].split() for w in ids]
    return data

def prepare_emb():
    print(20*"=", "load pre emb......", 20*"=")
    data = open('../data/pingan/char_embedding.txt', 'r').read().splitlines()
    data = {w.split()[0]:' '.join(w.split()[1:]) for w in data}
    return data


def build_wv(config, data):
    model = models.Word2Vec(data, size=300, window=10, min_count=1, workers=4)
    print(20*"=", "train model......", 20*"=")
    print("\t"*20)
    model.train(data, total_examples=len(data), epochs=10)
    words = (' '.join([' '.join(w) for w in data])).split()
    words = list(set(words))
    wv_out = prepare_emb()
    res = []
    for word in words:
        tmp = word + ' ' + ' '.join(['%.5f'% w for w in model.wv[word]])
        if word in wv_out:
            tmp += ' ' + wv_out[word]
        else:
            continue
        res.append(tmp)
    open(os.path.join(config['target_dir'], config['emb_file']), 'w').write('\n'.join(res))


def manage():
    config = get_config()
    if not os.path.exists(config['target_dir']):
        os.makedirs(config['target_dir'])
    print(20*"=", "load config...", 20*"=")
    print("\t"*2)
    print(20*"=", "load config...", 20*"=")
    ids = prepare_id(config)
    print(20*"=", "load id......", 20*"=")
    print("\t"*2)
    d = prepare_question(config)
    data = prepare_data(d, ids)
    build_wv(config, data)

if __name__ == '__main__':
    manage()
