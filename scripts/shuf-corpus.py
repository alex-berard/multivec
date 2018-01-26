#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import random
import argparse
import shutil

help_msg = """\
Shuffles a corpus.

Usage example:
    shuf-corpus.py data/my_corpus data/my_corpus.shuf fr en
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='name of the input corpus (path without extension, e.g. data/my_corpus)')
    parser.add_argument('--output', help='name of the output corpus (if not specified, input corpus is overwritten)')
    parser.add_argument('--seed', type=int)
    parser.add_argument('extensions', nargs='+', help='extensions (e.g. fr, en)')

    args = parser.parse_args()

    corpus = args.corpus

    if args.output is not None:
        output = args.output
    else:
        output = corpus

    input_files = ['{0}.{1}'.format(args.corpus, ext) for ext in args.extensions]
    output_files = ['{0}.{1}'.format(output, ext) for ext in args.extensions]

    # reads the whole contents into memory (might cause problems if the files are too large)
    # TODO: process files one by one
    contents = []
    for filename in input_files:
        with open(filename) as f:
            contents.append(f.readlines())

    indices = list(range(len(contents[0])))
    random.seed(args.seed)
    random.shuffle(indices)

    contents = [[content[i] for i in indices] for content in contents]

    for filename, content in zip(output_files, contents):
        with open(filename, 'w') as f:
            f.writelines(content)
