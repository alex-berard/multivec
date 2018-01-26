#!/usr/bin/env python3

from collections import Counter
import argparse
import sys
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('corpus')
parser.add_argument('-n', type=int)
parser.add_argument('--sort', action='store_true')

args = parser.parse_args()

with open(args.corpus) as f:
    vocab = Counter(w for line in f for w in line.split())

with open(args.model) as f:
    _, dim = next(f).split()
    if args.sort:
        lines = sorted(f, key=lambda line: -vocab[line.split(' ', 1)[0]])
        if args.n:
            lines = lines[:args.n]
    else:
        most_common = set(w for w, _ in vocab.most_common(args.n))
        lines = [line for line in f if line.split(' ', 1)[0] in most_common]

    print('{} {}'.format(len(lines), dim))
    for line in lines:
        print(line.strip())
