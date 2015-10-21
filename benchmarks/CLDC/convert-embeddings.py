#!/usr/bin/python2
from __future__ import division
import sys

if __name__ == '__main__':
    next(sys.stdin)
    for line in sys.stdin:
        line = line.strip()
        word, vector = line.split(' ', 1)
        print('{0} : {1}'.format(word, vector))
