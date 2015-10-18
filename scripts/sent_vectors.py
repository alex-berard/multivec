#!/usr/bin/env python2
import sys
sys.path.append('python-wrapper')
from word2vec import Word2vec

model_filename = sys.argv[1]
model = Word2vec()
model.load(model_filename)

for sent in sys.stdin:
    sent = sent.strip()
    if not sent:
        continue
    try:
        vec = model.sent_vec(sent)
    except:
        vec = [0.0] * 100
    print ' '.join(map(str, vec))
