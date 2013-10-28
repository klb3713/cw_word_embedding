# -*- coding: utf-8 -*-
__author__ = 'klb3713'

"""
Methods for getting samples.
"""


import string
import config


class TrainingMiniBatchStream(object):

    def __init__(self):
        self.count = 0

    def __iter__(self):
        self.filename = config.SAMPLE_FILE
        mini_batch = []
        for line in open(self.filename, 'r'):
            word_ids = [int(id) for id in string.strip(line).split()]
            self.count += 1
            mini_batch.append(word_ids)
            if len(mini_batch) >= config.MINIBATCH_SIZE:
                yield mini_batch
                mini_batch = []

