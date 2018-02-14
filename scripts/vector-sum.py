#!/usr/bin/python3
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vectors')
parser.add_argument('corpus')
parser.add_argument('--avg', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    vectors = dict()

    with open(args.vectors) as f:
        for line in f:
            w, v = line.split(' ', 1)
            vectors[w] = np.array(list(map(float, v.split())))

    dimension = len(next(iter(vectors.values())))

    with open(args.corpus) as f:
        for line in f:
            words = line.split()
            v = np.zeros(dimension)
            n = 0
            for word in words:
                if word in vectors:
                    n += 1
                    v += vectors[word]
            if n > 0 and args.avg:
                v /= n
            print(' '.join(map(str, v)))
