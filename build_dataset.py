# -*- coding: utf-8 -*-
__author__ = 'klb3713'

import os
import sys
import re
import config
import vocabulary

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

p_word = re.compile(r'[a-zA-Z0-9]+')


def getNormalWord(word):
    if word.isdigit():
        return config.SYMBOL_WORD
    elif word.isalnum():
        return word.lower()

    word = ''.join(p_word.findall(word))
    if word.isdigit():
        return config.SYMBOL_WORD
    else:
        return word.lower()


def build_vocabulary():
    if not os.path.exists(config.TRAIN_FILE):
        print("Error: can't find train file '%s'!" % config.TRAIN_FILE)
        exit()

    if config.INCLUDE_UNKNOWN_WORD:
        vocabulary.add(config.UNKNOWN_WORD, config.WORD_COUNT + 1)
    vocabulary.add(config.SYMBOL_WORD)
    vocabulary.add(config.PADDING_WORD, config.WORD_COUNT + 1)

    train_size = 0
    with open(config.TRAIN_FILE, 'r') as f:
        for line in f:
            line = line.strip('\n')
            words = line.split()
            for word in words:
                word = getNormalWord(word)
                if word:
                    train_size += 1
                    vocabulary.add(word)

    vocabulary.delete_word()
    vocabulary.save_vocabulary()
    print("TRAIN SIZE: %d" % train_size)
    print("VOCABULARY SIZE: %d" % vocabulary.length())
    vocabulary.dump_vocabulary()


def build_samples():
    if not vocabulary.length():
        vocabulary.load_vocabulary()

    sample_file = open(config.SAMPLE_FILE, 'w')
    for line in open(config.TRAIN_FILE, 'r'):
        line = line.strip('\n')
        words = [getNormalWord(word) for word in line.split() if word]
        word_ids = [vocabulary.id(word) for word in words]
        sent_length = len(word_ids)
        half_window = config.WINDOW_SIZE / 2
        window = []
        padding_id = vocabulary.id(config.PADDING_WORD)
        unknown_id = vocabulary.id(config.UNKNOWN_WORD)
        symbol_id = vocabulary.id(config.SYMBOL_WORD)
        for index, word_id in enumerate(word_ids):
            if word_id == unknown_id:
                continue
            if index - half_window >= 0 and index + half_window < sent_length:
                window = word_ids[index - half_window : index + half_window + 1]
                if window.count(unknown_id) + window.count(symbol_id) + window.count(padding_id) <= half_window:
                    sample_file.write(' '.join([str(id) for id in window]) + '\n')
                window = []
                continue

            if index - half_window < 0:
                for i in range(half_window - index):
                    window.append(padding_id)
                window.extend(word_ids[:index + 1])
            else:
                window.extend(word_ids[index - half_window : index + 1])

            if index + half_window >= sent_length:
                window.extend(word_ids[index + 1:])
                for i in range(index + half_window - sent_length + 1):
                    window.append(padding_id)
            else:
                window.extend(word_ids[index + 1 : index + half_window + 1])

            if window.count(unknown_id) + window.count(symbol_id) + window.count(padding_id) <= half_window:
                sample_file.write(' '.join([str(id) for id in window]) + '\n')
            window = []

    sample_file.close()


if __name__ == "__main__":
    # build_vocabulary()
    build_samples()