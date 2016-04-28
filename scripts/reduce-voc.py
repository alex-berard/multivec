#!/usr/bin/python2
from __future__ import division
import sys
import argparse
from operator import itemgetter
from collections import Counter


help_msg = '''\
Filter tokens in a corpus. In each corpus file, words that appear less than 
`min-count` times are replaced by an UNK symbol. In addition only the 
`max-vocab-size` most frequent tokens are kept (the other ones are replaced by UNK).\
'''


def get_vocab(filename, args):
    counts = Counter()
    with open(filename) as file_:
        for line in file_:
            for word in line.split():
                counts[word] += 1
    
    words = [(w, c) for w, c in counts.iteritems() if c >= args.min_count]
    
    if 0 < args.max_vocab_size < len(words):
        words = sorted(words, key=itemgetter(1), reverse=True)[:args.max_vocab_size]
    
    return set(w for w, _ in words)


def filter_vocab(input_filename, output_filename, vocab, args):
    with open(input_filename) as input_file, open(output_filename, 'w') as output_file:
        for line in input_file:
            line = ' '.join(w if w in vocab else args.unk_symbol for w in line.split())
            output_file.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='input corpus')
    parser.add_argument('output_corpus', help='output corpus')
    parser.add_argument('extensions', nargs='+', help='list of extensions')
    parser.add_argument('--min-count', type=int, default=0)
    parser.add_argument('--max-vocab-size', type=int, default=0)
    parser.add_argument('--unk-symbol', default='<UNK>')
    
    args = parser.parse_args()

    for ext in args.extensions:
        input_filename = '{}.{}'.format(args.corpus, ext)
        output_filename = '{}.{}'.format(args.output_corpus, ext)
        vocab = get_vocab(input_filename, args)
        filter_vocab(input_filename, output_filename, vocab, args)

