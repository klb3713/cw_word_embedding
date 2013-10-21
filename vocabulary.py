# -*- coding: utf-8 -*-
__author__ = 'klb3713'


"""
Automatically load the wordmap, if available.
"""


import cPickle
import config


def readWordMap():
    """
    Read the word ID map and return.
    """
    try:
        with open(config.VOCABULARY_IDMAP_FILE, 'r') as f:
            word_map = cPickle.load(f)
            word_map.str = word_map.key
            return word_map
    except Exception, e:
        print(e)


def writeWordMap(word_map):
    """
    Write the word ID map, passed as a parameter.
    """
    print "Writing word map to %s..." % config.VOCABULARY_IDMAP_FILE
    with open(config.VOCABULARY_IDMAP_FILE, 'w') as f:
        cPickle.dump(word_map, f)

word_map = readWordMap()