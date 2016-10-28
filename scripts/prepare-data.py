#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip, islice
from random import shuffle
from contextlib import contextmanager
from collections import Counter
from operator import itemgetter
import argparse
import subprocess
import tempfile
import os
import logging
import sys
import shlex
import shutil


help_msg = """\
Apply any number of those pre-processing steps to given corpus:
Tokenization, lowercasing, shuffling, filtering of lines according to length,
splitting into train/dev/test, punctuation and digit normalization.
"""

temporary_files = []


@contextmanager
def open_files(names, mode='r'):
    files = []
    try:
        for name_ in names:
            files.append(open(name_, mode=mode))
        yield files
    finally:
        for file_ in files:
            file_.close()


@contextmanager
def open_temp_files(num=1, mode='w', delete=False):
    files = []
    try:
        for _ in range(num):
            files.append(tempfile.NamedTemporaryFile(mode=mode, delete=delete))
            if not delete:
                temporary_files.append(files[-1].name)
        yield files
    finally:
        for file_ in files:
            file_.close()


def process_file(corpus, id_, args):
    filename = '{}.{}'.format(corpus, args.extensions[id_])
    logging.info('processing ' + filename)

    lang = args.lang[id_]

    with open_temp_files(num=1) as output_, open(filename) as input_:
        output_ = output_[0]

        def path_to(script_name):
            if args.scripts is None:
                return script_name
            else:
                return os.path.join(args.scripts, script_name)

        processes = [['cat']]   # just copy file if there is no other operation
        if args.normalize_punk:
            processes.append([path_to('normalize-punctuation.perl'), '-l',
                              lang])
            # replace html entities FIXME (doesn't seem to work)
            # processes.append(shlex.split("perl -MHTML::Entities -pe 'decode_entities($_);'"))
        if args.tokenize:
            processes.append([path_to('tokenizer.perl'), '-l', lang,
                              '-threads', str(args.threads)])
        if args.lowercase:
            processes.append([path_to('lowercase.perl')])
        if args.normalize_digits:
            processes.append(['sed', 's/[[:digit:]]/0/g'])

        ps = None
        for i, process in enumerate(processes):
            stdout = output_ if i == len(processes) - 1 else subprocess.PIPE
            stdin = input_ if i == 0 else ps.stdout

            ps = subprocess.Popen(process, stdin=stdin, stdout=stdout,
                                  stderr=open('/dev/null', 'w'))

        ps.wait()
        return output_.name


def process_corpus(corpus, args):
    input_filenames = [process_file(corpus, i, args)
                 for i in range(len(args.extensions))]

    with open_files(input_filenames) as input_files,\
         open_temp_files(len(input_filenames)) as output_files:

        # (lazy) sequence of sentence tuples
        all_lines = (lines for lines in izip(*input_files) if
                     all(args.min <= len(line.split()) <= args.max
                     for line in lines))

        if args.shuffle:
            all_lines = list(all_lines)  # not lazy anymore
            shuffle(all_lines)

        for lines in all_lines:  # keeps it lazy if no shuffle
            for line, output_file in zip(lines, output_files):
                output_file.write(line)

        return [f.name for f in output_files]


def split_corpus(filenames, sizes, args):
    with open_files(filenames) as input_files:
        output_filenames = []
    
        for size in sizes:
            if size == 0:
                output_filenames.append(None)
                continue
                
            with open_temp_files(num=len(args.extensions)) as output_files:
                for input_file, output_file in zip(input_files, output_files):
                    # If size is None, this will read the whole file.
                    # That's why we put train last.
                    output_file.writelines(islice(input_file, size))
                output_filenames.append([f.name for f in output_files])

        return output_filenames


def get_vocab(filename, args):
    with open(filename) as file_:
        counts = Counter(word for line in file_ for word in line.split())
    
    words = [(w, c) for w, c in counts.iteritems() if c >= args.min_count]
    
    max_vocab_size = args.max_vocab_size
    if 0 < max_vocab_size < len(words):
        words = sorted(words, key=itemgetter(1), reverse=True)[:max_vocab_size]
    
    return set(w for w, _ in words)


def move_and_filter(filenames, output_corpus, args, vocabs=None):
    output_filenames = ['{}.{}'.format(output_corpus, ext)
                        for ext in args.extensions]

    if not vocabs:
        for filename, output_filename in zip(filenames, output_filenames):
            shutil.move(filename, output_filename)
        return
    
    for filename, output_filename, vocab in zip(filenames, output_filenames,
                                                vocabs):
        with open(filename) as input_file,\
             open(output_filename, 'w') as output_file:
        
            for line in input_file:
                line = ' '.join(w if w in vocab else args.unk_symbol
                                for w in line.split())
                
                output_file.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='training corpus')
    parser.add_argument('output_corpus', help='destination corpus')
    parser.add_argument('extensions', nargs='+', help='list of extensions')
    parser.add_argument('--dev-corpus', help='development corpus')
    parser.add_argument('--test-corpus', help='test corpus')
    parser.add_argument('--scripts', help='path to script directory '
                        '(None if in $PATH)', default='scripts')
    parser.add_argument('--dev-size', type=int,
                        help='size of development corpus', default=0)
    parser.add_argument('--test-size', type=int,
                        help='size of test corpus', default=0)
    parser.add_argument('--train-size', type=int,
                        help='size of training corpus (default: maximum)')
    parser.add_argument('--lang', nargs='+', help='optional list of language '
                        'codes (when different than file extensions)')
    parser.add_argument('--normalize-punk', help='normalize punctuation',
                        action='store_true')
    parser.add_argument('--normalize-digits', help='normalize digits '
                        '(replace all digits with 0)', action='store_true')
    parser.add_argument('--lowercase', help='put everything to lowercase',
                        action='store_true')
    parser.add_argument('--shuffle', help='shuffle the corpus',
                        action='store_true')
    parser.add_argument('--tokenize', dest='tokenize',
                        help='tokenize the corpus', action='store_true')
    parser.add_argument('-v', '--verbose', help='verbose mode',
                        action='store_true')
    parser.add_argument('--min', type=int, default=1,
                        help='min number of tokens per line')
    parser.add_argument('--max', type=int, default=0,
                        help='max number of tokens per line (0 for no limit)')
    parser.add_argument('--threads', type=int, default=16,
                        help='number of threads for tokenizer')
    parser.add_argument('--min-count', type=int, default=0)
    parser.add_argument('--max-vocab-size', type=int, default=0)
    parser.add_argument('--unk-symbol', default='<UNK>')

    args = parser.parse_args()

    args.max = args.max if args.max > 0 else float('inf')

    if args.lang is None:
        args.lang = args.extensions
    elif len(args.lang) != len(args.extensions):
        sys.exit('wrong number of values for parameter --lang')

    if args.verbose:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    
    output_dir = os.path.dirname(args.output_corpus)
    if output_dir and not os.path.exists(output_dir):
        logging.info('creating directory')
        os.makedirs(output_dir)

    try:
        input_corpora = (args.dev_corpus, args.test_corpus, args.corpus)
        output_corpora = (args.output_corpus + '.dev' ,
                          args.output_corpus + '.test',
                          args.output_corpus)
        
        # list of temporary files for each corpus (dev, test, train)
        # a value of None means no such corpus
        filenames = [None, None, None]
        for i, corpus in enumerate(input_corpora):
            if corpus is not None:
                filenames[i] = process_corpus(corpus, args)
        
        # split files
        sizes = [
            args.dev_size if not args.dev_corpus else 0,
            args.test_size if not args.test_corpus else 0,
            args.train_size
        ]
        
        if any(sizes):
            logging.info('splitting files')
            # returns a list in the same format as `filenames`
            split_filenames = split_corpus(filenames[-1], sizes, args)
            
            # union of `filenames` and `split_filenames`
            for i, filenames_ in enumerate(split_filenames):
                if filenames_ is not None:
                    filenames[i] = filenames_
        
        vocabs = None
        if args.max_vocab_size or args.min_count:
            # vocabularies are created from training corpus
            vocabs = [get_vocab(filename, args) for filename in filenames[-1]]
            
        # move temporary files to their destination
        for filenames_, output_corpus in zip(filenames, output_corpora):
            if filenames_ is not None:
                move_and_filter(filenames_, output_corpus, args, vocabs)
        
    finally:
        logging.info('removing temporary files')
        for name in temporary_files:
            try:
                os.remove(name)
            except OSError:
                pass

