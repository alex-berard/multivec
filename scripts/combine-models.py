#!/usr/bin/env python3

from collections import OrderedDict
import argparse
import sys
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
parser.add_argument('--output')
parser.add_argument('--concat')

args = parser.parse_args()

model = OrderedDict()
vocab = None

for filename in args.models:
    print('reading {}'.format(filename))
    with open(filename) as f:
        vocab_ = set()
        next(f)
        
        for line in f:
            word, *vec = line.split()
            vec = np.array([float(x) for x in vec])
            if word not in vocab_:
                vocab_.add(word)
                model[word] = model.get(word, 0) + vec
        
        vocab = vocab_ if vocab is None else vocab.intersection(vocab_)

model = [(w, vec) for w, vec in model.items() if w in vocab]

output_file = open(args.output, 'w') if args.output else sys.stdout

output_file.write('{} {}\n'.format(len(model), len(model[0][1])))
for w, vec in model:
    output_file.write(' '.join([w] + [str(x) for x in vec]) + '\n')
    
output_file.close()