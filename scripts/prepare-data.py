#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip, islice
from random import shuffle
from contextlib import contextmanager
import argparse
import subprocess
import tempfile
import os
import logging
import sys
import shlex


help_msg = """\
Apply any number of those pre-processing steps to given corpus:
Tokenization, lowercasing, shuffling, filtering of lines according to length, splitting into train/dev/test,
punctuation normalization and digit normalization.
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
        if args.tokenize:
            processes.append([path_to('tokenizer.perl'), '-l', lang, '-threads',
                              str(args.threads)])
        if args.lowercase:
            processes.append([path_to('lowercase.perl')])
        if args.normalize_numbers:
            processes.append(['sed', 's/[[:digit:]]/0/g'])

        ps = None
        for i, process in enumerate(processes):
            stdout = output_ if i == len(processes) - 1 else subprocess.PIPE
            stdin = input_ if i == 0 else ps.stdout

            ps = subprocess.Popen(process, stdin=stdin, stdout=stdout,
                                  stderr=open('/dev/null', 'w'))

        ps.wait()
        return output_.name


def process_corpus(corpus, args, output_corpus=None):
    input_filenames = [process_file(corpus, i, args)
                 for i in range(len(args.extensions))]

    output_filenames = None
    if output_corpus is not None:
        output_filenames = ['{}.{}'.format(output_corpus, ext)
                            for ext in args.extensions]

    with open_files(input_filenames) as input_files,\
            (open_temp_files(len(input_filenames)) if not output_filenames
            else open_files(output_filenames, 'w')) as output_files:

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


def split_corpus(filenames, dest_corpora, extensions):
      
    with open_files(filenames) as input_files:
        for corpus, size in reversed(dest_corpora):  # puts train corpus last
            if size != 0:
                output_filenames = ['{}.{}'.format(corpus, ext)
                                    for ext in extensions]
                with open_files(output_filenames, mode='w') as output_files:
                    for input_file, output_file in zip(input_files, output_files):
                        output_file.writelines(islice(input_file, size))
                        # If size is None, this will read the whole file.
                        # That's why we put train last.

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='training corpus')
    parser.add_argument('output_corpus',
                        help='directory where the files will be copied')
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
                        help='size of training corpus (defaults to maximum)')

    parser.add_argument('--lang', nargs='+', help='optional list of language '
                                                  'codes (when different '\
                                                  'than file extensions)')

    parser.add_argument('--normalize-punk', help='normalize punctuation',
                        action='store_true')
    parser.add_argument('--normalize-numbers', help='normalize numbers '
                        '(replace all digits with 0)', action='store_true')
    parser.add_argument('--lowercase', help='put everything to lowercase',
                        action='store_true')
    parser.add_argument('--shuffle', help='shuffle the corpus',
                        action='store_true')
    parser.add_argument('--tokenize', dest='tokenize',
                        help='toggle tokenization', action='store_true')

    parser.add_argument('-v', '--verbose', help='verbose mode',
                        action='store_true')

    parser.add_argument('--min', type=int, default=1,
                        help='min number of tokens per line')
    parser.add_argument('--max', type=int, default=50,
                        help='max number of tokens per line (0 for no limit)')

    parser.add_argument('--threads', type=int, default=16)

    args = parser.parse_args()

    args.max = args.max if args.max > 0 else float('inf')

    if args.lang is None:
        args.lang = args.extensions
    elif len(args.lang) != args.extensions:
        sys.exit('wrong number of values for parameter --lang')

    if args.verbose:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    
    output_dir = os.path.dirname(args.output_corpus)
    if output_dir and not os.path.exists(output_dir):
        logging.info('creating directory')
        os.makedirs(output_dir)

    output_train = args.output_corpus
    output_test = args.output_corpus + '.test'
    output_dev = args.output_corpus + '.dev'

    try:
        if args.dev_corpus:
            logging.info('processing dev corpus')
            process_corpus(args.dev_corpus, args, output_dev)
        if args.test_corpus:
            logging.info('processing test corpus')
            process_corpus(args.test_corpus, args, output_test)

        logging.info('processing train corpus')
        if args.dev_corpus and args.test_corpus:
            process_corpus(args.corpus, args, output_train)
        else:
            filenames = process_corpus(args.corpus, args)                
            dest_corpora = [(output_train, args.train_size)]
            if not args.test_corpus:
                dest_corpora.append((output_test, args.test_size))
            if not args.dev_corpus:
                dest_corpora.append((output_dev, args.dev_size))

            logging.info('splitting files')
            split_corpus(filenames, dest_corpora, args.extensions)

    finally:
        logging.info('removing temporary files')
        for name in temporary_files:  # remove temporary files
            try:
                os.remove(name)
            except OSError:
                pass
