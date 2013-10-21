# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import vocabulary
import config
import common.idmap


p_word = re.compile(r'\w+')

def getNormalWord(word):
    if word.isalnum():
        return word.lower()
    w = p_word.search(word)
    if w:
        return w.group().lower()
    else:
        return None


if __name__ == "__main__":
    words = []
    with open(config.VOCABULARY, 'r') as f:
        for line in f:
            line = line.strip('\n')
            w = line.split()[-1]
            w = getNormalWord(w)
            if w and w not in words:
                words.append(w)

    v = common.idmap.IDmap(words, allow_unknown=False)
    config.VOCABULARY_SIZE = v.len
    print("VOCABULARY_SIZE: ", config.VOCABULARY_SIZE)
    vocabulary.writeWordMap(v)
