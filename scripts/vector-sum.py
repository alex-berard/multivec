#!/usr/bin/python2
from __future__ import division
import sys
import numpy as np

if __name__ == '__main__':
    vectors_filename, corpus = sys.argv[1:]
    vectors = dict()

    with open(vectors_filename) as f:
        for line in f:
            w, v = line.split(' ', 1)
            vectors[w] = np.array(map(float, v.split()))

    dimension = len(next(vectors.itervalues()))

    with open(corpus) as f:
        for line in f:
            words = line.split()
            v = np.zeros(dimension)
            n = 0
            for word in words:
                if word in vectors:
                    n += 1
                    v += vectors[word]
            if n > 0:
                v /= n
            print(' '.join(map(str, v)))
