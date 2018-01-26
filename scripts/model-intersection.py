#!/usr/bin/env python3

import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+')
parser.add_argument('--output-dir')
parser.add_argument('--output-suffix')
parser.add_argument('--questions')
parser.add_argument('--lower', action='store_true')

args = parser.parse_args()

if not args.output_dir and not args.output_suffix:
    print('error: at least one of --output-dir and --output-suffix should be set')
    sys.exit()

vocab = None
vocab_union = set()

vocabs = []

def transform(word):
    if args.lower:
        return word.lower()
    else:
        return word

for model in args.models:
    print('reading {}'.format(model))
    with open(model) as f:
        next(f)
        vocab_ = set(transform(line.split()[0]) for line in f)
        vocab = vocab_ if vocab is None else vocab.intersection(vocab_)
        vocabs.append(vocab_)
        vocab_union = vocab_union.union(vocab_)

print('Intersection size: {}'.format(len(vocab)))
print('Union size: {}'.format(len(vocab_union)))

def get_output_filename(filename):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.join(args.output_dir, os.path.basename(filename))
    if args.output_suffix:
        filename += args.output_suffix
    return filename

if args.questions:
    question_vocab = set()
    question_count = 0
    with open(args.questions) as questions:
        for line in questions:
            if not line.startswith(':'):
                words = [transform(word) for word in line.split()]
                question_vocab.update(words)
                if all(w in vocab for w in words):
                    question_count += 1
    print('Total questions: {}'.format(question_count))
    
    # output_filename = get_output_filename(args.questions)
    # assert output_filename != args.questions
    # print('{} => {}'.format(args.questions, output_filename))
    # with open(args.questions) as questions, open(output_filename, 'w') as output_file:
    #     for line in questions:
    #         if line.startswith(':') or all(w in vocab for w in line.split()):
    #             output_file.write(line)

for model in args.models:
    output_filename = get_output_filename(model)
    assert output_filename != model
    print('{} => {}'.format(model, output_filename))
    
    with open(model) as input_file, open(output_filename, 'w') as output_file:
        _, dim = next(input_file).split()
        output_file.write('{} {}\n'.format(len(vocab), dim))
        vocab_ = set()
        for line in input_file:
            word, vec = line.split(' ', 1)
            word = transform(word)
            if word in vocab and word not in vocab_:
                output_file.write(word + ' ' + vec)
                vocab_.add(word)
