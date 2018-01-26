#!/usr/bin/env python3
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
parser.add_argument('--lower', action='store_true')
#parser.add_argument('questions')

args = parser.parse_args()

def transform(word):
    if args.lower:
        return word.lower()
    else:
        return word

vocabs = []
for model in args.models:
    vocab = set()
    with open(model) as f:
        next(f)
        for line in f:
            word, _ = line.split(' ', 1)
            vocab.add(transform(word))
    vocabs.append(vocab)

vocab = set.intersection(*vocabs)

for question in sys.stdin:
    if question.startswith(':'):
        sys.stdout.write(question)
        continue
    words = question.split()
    if all(transform(word) in vocab for word in words):
        sys.stdout.write(question)

