import random
import os
import pickle
import math
import numpy as np
import pickle

def loadGloVe(filename, vocab_exist=None):
    vocab = []
    vocab_dict = {}
    embd = []
    with open(filename, 'r') as fin:
        for line in fin:
            row = line.strip().split(' ')
            if vocab_exist is None or row[0] in vocab_exist:
                vocab.append(row[0])
                vocab_dict[row[0]] = len(vocab) - 1
                embd.append(row[1:])
        print('Loaded GloVe!')
    embd = np.array(embd)
    return vocab, vocab_dict, embd

def build_vocab(dataset, pretrained_embeddings_path):
    vocab = None
    if os.path.isfile('{}/vocab_python2.pkl'.format(dataset)):
        print('loading saved vocab...')
        with open('{}/vocab_python2.pkl'.format(dataset), 'rb') as fin:
            vocab = pickle.load(fin)
    else:
        code = int(0)
        vocab = {}
        vocab['UNKNOWN'] = code
        code += 1
        filenames = ['{}/train.data'.format(dataset), '{}/valid.data'.format(dataset), '{}/test.data'.format(dataset)]
        for filename in filenames:
            for line in open(filename):
                items = line.strip().split(' ')
                for i in range(1, len(items)):
                    words = items[i].split('_')
                    for word in words:
                        if word not in vocab:
                            vocab[word] = code
                            code += 1
    embd = None
    print("#vocab,",len(vocab))
    print(vocab['isotopy'])
    if os.path.isfile('{}/embd_python2.pkl'.format(dataset)):
        print('loading saved embd...')
        with open('{}/embd_python2.pkl'.format(dataset), 'rb') as fin:
            embd = pickle.load(fin)
    elif len(pretrained_embeddings_path) > 0:
        vocab_all, vocab_dict_all, embd_all = loadGloVe(pretrained_embeddings_path, vocab)
        embd = []
        for k, v in vocab.items():
            try:
                index = vocab_dict_all[k]
                embd.append(embd_all[index])
            except:
                embd.append(np.random.uniform(-0.05, 0.05, (embd_all.shape[1])))
        embd = np.array(embd)
    return vocab, embd

if __name__=="__main__":
    vocab, embd = build_vocab("../data/mcql_processed","../data/data_embeddings/glove.840B.300d.txt")
    output = open('../data/mcql_processed/vocab_python2.pkl', 'wb')
    pickle.dump(vocab, output)
    output.close()
    print("dump to vocab.pkl!")
    output = open('../data/mcql_processed/embd_python2.pkl', 'wb')
    pickle.dump(embd, output)
    output.close()
    print("dump to embd.pkl!")
    print(len(vocab))
    L = "If we observe a pebble in a pool".lower().split()
    for i in L:
        print("find, ",i)
        print(embd[vocab[i]])