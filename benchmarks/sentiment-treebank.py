#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('sentences')
parser.add_argument('split')
parser.add_argument('labels')
parser.add_argument('dictionary')
parser.add_argument('segmentation')

parser.add_argument('output_dir')

args = parser.parse_args()

train_sentences = []
dev_sentences = []
test_sentences = []

sentences = defaultdict(list)
all_segments = defaultdict(list)
dictionary = dict()

with open(args.sentences) as sentence_file, open(args.split) as split_file,\
     open(args.labels) as label_file, open(args.dictionary) as dict_file,\
     open(args.segmentation) as seg_file:
    next(sentence_file)
    next(split_file)
    next(label_file)
    
    labels = []
    for i, line in enumerate(label_file):
        index, label = line.split('|', 1)
        assert int(index) == i
        labels.append(float(label))
    for line in dict_file:
        segment, index = line.split('|', 1)
        dictionary[segment] = labels[int(index)]
    
    for split, segments in zip(split_file, seg_file):
        _, set_id = split.split(',', 1)
        segments = segments.strip().split('|')
        sentence = ' '.join(segments)
        set_id = int(set_id)
        label = dictionary[sentence]
        sentences[set_id].append((sentence, label))
        for segment in segments:
            label = dictionary[segment]
            all_segments[set_id].append((segment, label))

corpora = ['train', 'test', 'dev']
os.makedirs(args.output_dir, exist_ok=True)

for corpus_id, corpus in enumerate(corpora, 1):
    sentences_ = sentences[corpus_id]
    segments = all_segments[corpus_id]
    
    filename = os.path.join(args.output_dir, corpus)
    for filename, data in zip([filename + '_sentences', filename + '_segments'], [sentences[corpus_id], all_segments[corpus_id]]):
    
        with open(filename, 'w') as data_file, open(filename + '.labels', 'w') as label_file:
            for sentence, label in data:
                data_file.write(sentence + '\n')
                label_file.write(str(label) + '\n')
