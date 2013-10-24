# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import os
import cPickle
import config


word_to_id = {}
words = []


def add(word):
    """
        Add word to vocabulary and return its id.
        return -1 if this word is exist.
    """
    if word not in word_to_id:
        word_to_id[word] = len(words)
        words.append([word, 1])
        return word_to_id[word]
    else:
        words[word_to_id[word]][1] += 1
        return -1


def exists(word):
    """ Return True iff this key is in the map, or if self.allow_unknown is True """
    return word in word_to_id or config.INCLUDE_UNKNOWN_WORD


def id(word, add_if_word_doesnt_exist=False):
    """ Get the ID for this string. """
    if word in word_to_id:
        return word_to_id[word]
    elif add_if_word_doesnt_exist:
        add(word)
    elif config.INCLUDE_UNKNOWN_WORD:
        return word_to_id[config.UNKNOWN_WORD]
    else:
        return -1


def word(id):
    """ Get the word for this ID. """
    return words[id][0]


def word_freq(id):
    """ Get the word frequency for this ID. """
    return words[id][1]


def length():
    return len(words)


def load_vocabulary():
    """ Load the words and word_to_id. """

    global word_to_id
    global words

    if not os.path.exists(config.VOCABULARY_FILE):
        print("Error: can't find vocabulary dump file '%s'!" % config.VOCABULARY_FILE)
        exit()
    print("Loading vocabulary from %s..." % config.VOCABULARY_FILE)
    with open(config.VOCABULARY_FILE, 'rb') as f:
        word_to_id = cPickle.load(f)
        words = cPickle.load(f)


def dump_vocabulary():
    """ Write the word ID map, passed as a parameter. """
    print("Writing vocabulary to %s..." % config.VOCABULARY_FILE)
    with open(config.VOCABULARY_FILE, 'wb') as f:
        cPickle.dump(word_to_id, f)
        cPickle.dump(words, f)
