# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import os
import cPickle
import logging
import config

logger = logging.getLogger(__name__)

word_to_id = {}
words = []


def add(word, count=1):
    """
        Add word to vocabulary and return its id.
        return -1 if this word is exist.
    """
    if word not in word_to_id:
        word_to_id[word] = len(words)
        words.append([word, count])
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


def read_vocabulary():
    """ Load the words and word_to_id. """

    global word_to_id
    global words

    if not os.path.exists(config.VOCABULARY_FILE):
        logger.info("Error: can't find vocabulary dump file '%s'!" % config.VOCABULARY_FILE)
        exit()

    logger.info("Loading vocabulary from %s..." % config.VOCABULARY_FILE)
    with open(config.VOCABULARY_FILE, 'r') as f:
        for line in f:
            word, count = line.strip('\n').split()
            words.append([word, int(count)])

    for id, word in enumerate(words):
        word_to_id[word[0]] = id


def load_vocabulary():
    """ Load the words and word_to_id. """

    global word_to_id
    global words

    if not os.path.exists(config.VOCABULARY_FILE):
        logger.info("Error: can't find vocabulary dump file '%s'!" % config.VOCABULARY_FILE)
        exit()

    if config.VOCABULARY_FILE.endswith('.txt'):
        read_vocabulary()
        return

    logger.info("Loading vocabulary from %s..." % config.VOCABULARY_FILE)
    with open(config.VOCABULARY_FILE, 'rb') as f:
        words = cPickle.load(f)
    for id, word in enumerate(words):
        word_to_id[word[0]] = id


def dump_vocabulary():
    """ Write the word ID map, passed as a parameter. """
    logger.info("Writing vocabulary to %s..." % config.VOCABULARY_FILE)
    with open(config.VOCABULARY_FILE, 'wb') as f:
        cPickle.dump(words, f)


def save_vocabulary():
    with open(config.SAVE_VOCABULARY, 'w') as voc_file:
        for word in words:
            voc_file.write("%s %d\n" % (word[0], word[1]))


def delete_word():
    global words
    words = filter(lambda x: x[1] > config.WORD_COUNT, words)


if __name__ == "__main__":
    load_vocabulary()
    save_vocabulary()
