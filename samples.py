# -*- coding: utf-8 -*-
__author__ = 'klb3713'

"""
Methods for getting samples.
"""

import sys
import string
import config
import vocabulary


class TrainingExampleStream(object):
    def __init__(self):
        self.count = 0
        pass
    
    def __iter__(self):
        self.filename = config.TRAIN_SENTENCES
        self.count = 0
        for l in open(self.filename, 'r'):
            prevwords = []
            for w in string.split(l):
                w = string.strip(w)
                id = None
                if vocabulary.exists(w):
                    prevwords.append(vocabulary.id(w))
                    if len(prevwords) >= config.WINDOW_SIZE:
                        self.count += 1
                        yield prevwords[:]
                else:
                    prevwords = []

    def __getstate__(self):
        return self.filename, self.count

    def __setstate__(self, state):
        """
        @warning: We ignore the filename.  If we wanted
        to be really fastidious, we would assume that
        HYPERPARAMETERS["TRAIN_SENTENCES"] might change.  The only
        problem is that if we change filesystems, the filename
        might change just because the base file is in a different
        path. So we issue a warning if the filename is different from
        """
        filename, count = state
        print >> sys.stderr, ("__setstate__(%s)..." % `state`)
        iter = self.__iter__()
        while count != self.count:
            iter.next()
        if self.filename != filename:
            assert self.filename == config.TRAIN_SENTENCES
            print >> sys.stderr, ("self.filename %s != filename given to __setstate__ %s" % (self.filename, filename))
        print >> sys.stderr, ("...__setstate__(%s)" % `state`)


class TrainingMinibatchStream(object):
    def __iter__(self):
        minibatch = []
        self.get_train_example = TrainingExampleStream()
        for e in self.get_train_example:
            minibatch.append(e)
            if len(minibatch) >= config.MINIBATCH_SIZE:
                yield minibatch
                minibatch = []

    def __getstate__(self):
        return (self.get_train_example.__getstate__(),)

    def __setstate__(self, state):
        """
        @warning: We ignore the filename.
        """
        self.get_train_example = TrainingExampleStream()
        self.get_train_example.__setstate__(state[0])


def get_validation_example():
    for l in open(config.VALIDATION_SENTENCES):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            if vocabulary.exists(w):
                prevwords.append(vocabulary.id(w))
                if len(prevwords) >= config.WINDOW_SIZE:
                    yield prevwords[:]
            else:
                prevwords = []
