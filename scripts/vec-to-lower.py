#!/usr/bin/env python3
import argparse
import sys

vocab = set()
lines = []

_, dim = next(sys.stdin).split()

for line in sys.stdin:
    word, vec = line.split(' ', 1)
    word = word.lower()
    if word not in vocab:
        vocab.add(word)
        lines.append(word + ' ' + vec)

print('{} {}'.format(len(lines), dim))
for line in lines:
    sys.stdout.write(line)

