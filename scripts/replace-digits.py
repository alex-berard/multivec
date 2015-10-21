#!/usr/bin/python2
from __future__ import division
import sys
import re

if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        print(re.sub(r'\d', '0', line))
