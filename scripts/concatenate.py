#!/usr/bin/env python3
import sys
import numpy as np
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
parser.add_argument('--sum', action='store_true')
parser.add_argument('--avg', action='store_true')
parser.add_argument('--lower', action='store_true')

args = parser.parse_args()

vectors = OrderedDict()

vocab = None

for model in args.models:
    vocab_ = set()
    with open(model) as f:
        next(f)
        for line in f:
            word, vec = line.split(' ', 1)
            vec = np.array(list(map(float, vec.split())))
            
            if args.lower:
                word = word.lower()
            
            vocab_.add(word)
            
            if word not in vectors:
                vectors[word] = vec
            elif args.sum or args.avg:
                vectors[word] += vec
            else:
                vectors[word] = np.concatenate([vectors[word], vec])

        if vocab is None:
            vocab = vocab_
        else:
            vocab = vocab.intersection(vocab_)

if args.avg:
    for key in vectors:
        vectors[key] /= len(args.models)

vectors = [(word, vec) for word, vec in vectors.items() if word in vocab]
dim = len(vectors[0][1])

print('{} {}'.format(len(vectors), dim))
for word, vec in vectors:
    print(' '.join([word] + [str(x) for x in vec]))

