#!/usr/bin/env python2
from __future__ import division
import argparse
import numpy
import os
import random
import sys

try:
    import multivec
except ImportError:
    sys.exit("Cannot import multivec. Please add `multivec.so` to the Python PATH, or copy it into this directory.")


class LabeledDocument(object):
    def __init__(self, label, repr):
        self.repr = repr
        self.label = label

    def __repr__(self):
        s = ' '.join('{0}:{1}'.format(i, v) for i, v in enumerate(self.repr, start=1))
        return '{0} {1}'.format(self.label, s)


def document_representation(label, filename, model):
    dimension = model.dimension()

    with open(filename) as f:
        document = ' '.join(f.read().split()).lower()
        try:
            vec = model.sent_vec(document)
        except ValueError:
            vec = numpy.zeros((dimension,))

    return LabeledDocument(label, vec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-dir', required=True)
    parser.add_argument('--model-file', required=True)     # multivec model
    parser.add_argument('--output-file', required=True)

    args = parser.parse_args()

    model_filename = args.model_file
    output_filename = args.output_file

    model = multivec.MonoModel()
    model.load(model_filename)

    labels = sorted(os.listdir(args.text_dir))
    label_ids = dict((v, i) for i, v in enumerate(labels, start=1))

    documents = []
    
    for label in labels:
        label_id = label_ids[label]
        dirname = os.path.join(args.text_dir, label)
        if not os.path.isdir(dirname):
            continue
        for filename in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, filename)
            document = document_representation(label_id, path, model)
            documents.append(document)
            
            if len(documents) % 1000 == 0:
                print len(documents)

    random.shuffle(documents)

    with open(output_filename, 'w') as f:
        f.write('\n'.join(map(str, documents)))
