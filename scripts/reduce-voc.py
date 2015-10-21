#!/usr/bin/python2
from __future__ import division
import sys
from collections import defaultdict

counter = defaultdict(int)

if __name__ == '__main__':
    filename = sys.argv[1]

    with open(filename) as f:
        for line in f:
            for word in line.split():
                counter[word] += 1

    with open(filename) as f:
        for line in f:
            words = line.split()
            print(' '.join(word if counter[word] >= 5 else '<unk>' for word in line.split()))
