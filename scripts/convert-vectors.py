#!/usr/bin/python2
from __future__ import division
import sys
import numpy as np
import argparse
from io import BufferedReader, FileIO
from itertools import takewhile, count


help_msg = """convert word embeddings files between text and binary formats"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg)

    parser.add_argument('filename', help='input file')
    parser.add_argument('output', help='output file')
    parser.add_argument('convert_to', choices=['bin', 'txt'])
    
    args = parser.parse_args()
    
    embeddings = []
    
    if args.convert_to == 'txt':  # then format must be bin
        with FileIO(args.filename, 'rb') as f:
            reader = BufferedReader(f)
            vocab_size, dimension = map(int, f.readline().split())
            for _ in range(vocab_size):
                w = ''.join(takewhile(lambda x: x != ' ', (reader.read(1) for _ in count())))
                s = reader.read(4 * dimension)
                reader.read(1)  # end of line character
                arr = np.fromstring(s, dtype=np.float32)
                embeddings.append((w, arr))
            assert not reader.peek(1)
    else:
        with open(args.filename) as f:
            vocab_size, dimension = map(int, f.readline().split())
            for line in f:
                w, s = line.strip().split(' ', 1)
                arr = np.fromstring(s, dtype=np.float32, sep=' ')
                embeddings.append((w, arr))
            assert len(embeddings) == vocab_size

    if args.convert_to == 'txt':
        with open(args.output, 'w') as f:
            f.write('{0} {1}\n'.format(vocab_size, dimension))
            f.writelines('{0} {1} \n'.format(w, ' '.join(map(str, vec))) for w, vec in embeddings)
    else:
        with open(args.output, 'wb') as f:
            # FIXME: seems to work, but not exactly same format as MultiVec
            f.write('{0} {1}\n'.format(vocab_size, dimension))
            f.writelines(b'{0} {1}\n'.format(w, vec.tostring()) for w, vec in embeddings)

