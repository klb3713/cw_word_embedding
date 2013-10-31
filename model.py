# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import math
import logging
import random
import copy
import numpy
import theano
import config
import vocabulary

from theano import tensor as T
from theano.sandbox.softsign import softsign
from parameters import Parameters

logger = logging.getLogger(__name__)


class Model(object):
    """
    A Model can:

    @type parameters: L{Parameters}
    @todo: Document
    """

    def __init__(self):
        vocabulary.load_vocabulary()
        self.parameters = Parameters()
        self.train_loss = 0
        self.train_err = 0
        self.train_lossnonzero = 0
        self.train_cnt = 0
        self.COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')

        self.train_function = self._get_train_function()

    def reset(self):
        self.train_loss = 0
        self.train_err = 0
        self.train_lossnonzero = 0
        self.train_cnt = 0

    def __getstate__(self):
        return (self.parameters,
                self.train_loss,
                self.train_err,
                self.train_lossnonzero,
                self.train_cnt)

    def __setstate__(self, state):
        (self.parameters,
         self.train_loss,
         self.train_err,
         self.train_lossnonzero,
         self.train_cnt) = state

    def _get_train_function(self):
        floatX = theano.config.floatX
        correct_inputs = T.matrix(dtype=floatX)

        noise_inputs = T.matrix(dtype=floatX)
        learning_rate = T.scalar(dtype=floatX)

        correct_prehidden = T.dot(correct_inputs, self.parameters.hidden_weights) + self.parameters.hidden_biases
        hidden = softsign(correct_prehidden)
        correct_score = T.dot(hidden, self.parameters.output_weights) + self.parameters.output_biases

        noise_prehidden = T.dot(noise_inputs, self.parameters.hidden_weights) + self.parameters.hidden_biases
        hidden = softsign(noise_prehidden)
        noise_score = T.dot(hidden, self.parameters.output_weights) + self.parameters.output_biases

        unpenalized_loss = T.clip(1 - correct_score + noise_score, 0, 1e999)
        total_loss = T.sum(unpenalized_loss)
        (doutput_weights, doutput_biases) = T.grad(total_loss,
                                  [self.parameters.output_weights, self.parameters.output_biases])

        dcorrect_inputs = T.grad(total_loss, correct_inputs)
        dnoise_inputs = T.grad(total_loss, noise_inputs)

        para_gpara = zip((self.parameters.output_weights, self.parameters.output_biases),
                         (doutput_weights, doutput_biases))
        updates = [(p, p - learning_rate * gp) for p, gp in para_gpara]

        logger.info("About to compile train function...")
        train_function = theano.function([correct_inputs, noise_inputs, learning_rate],
                                         [theano.Out(theano.sandbox.cuda.basic_ops.gpu_from_host(para), borrow=True)
                                          for para in [dcorrect_inputs, dnoise_inputs, total_loss,
                                                       unpenalized_loss, correct_score, noise_score]],
                                         mode=self.COMPILE_MODE,
                                         updates=updates)
        logger.info("Done constructing function for train")
        return train_function

    def embeds(self, sequences):
        """
        Embed sequences of vocabulary IDs.
        If we are given a list of MINIBATCH lists of SEQLEN items,
        return a matrices of shape (MINIBATCH, EMBSIZE*SEQLEN)
        """
        embs = []
        for sequence in sequences:
            seq = [self.parameters.embeddings[s] for s in sequence]
            embs.append(numpy.concatenate(seq))
        return numpy.vstack(embs)

    def corrupt_examples(self, correct_sequences):
        noise_sequences = []
        weights = []
        half_window = config.WINDOW_SIZE / 2

        for e in correct_sequences:
            noise_sequence = copy.copy(e)
            noise_sequence[half_window] = random.randint(0, self.parameters.vocab_size - 1)
            weight = self.parameters.vocab_size
            noise_sequences.append(noise_sequence)
            weights.append(weight)

        return noise_sequences, weights

    def train(self, correct_sequences):
        learning_rate = config.LEARNING_RATE
        noise_sequences, weights = self.corrupt_examples(correct_sequences)

        r = self.train_function(self.embeds(correct_sequences), self.embeds(noise_sequences), learning_rate * weights[0])
        (dcorrect_inputss, dnoise_inputss, total_loss, losses, correct_scores, noise_scores) = \
            [numpy.asarray(output) for output in r]

        self.train_loss += total_loss
        self.train_err += (correct_scores <= noise_scores).sum()
        self.train_lossnonzero += (losses > 0).sum()

        for index in range(len(correct_sequences)):
            correct_sequence = correct_sequences[index]
            noise_sequence = noise_sequences[index]

            dcorrect_inputs = dcorrect_inputss[index].reshape((config.WINDOW_SIZE, config.EMBEDDING_SIZE))
            dnoise_inputs = dnoise_inputss[index].reshape((config.WINDOW_SIZE, config.EMBEDDING_SIZE))

            # embedding_learning_rate = config.EMBEDDING_LEARNING_RATE * weights[0]
            embedding_learning_rate = config.LEARNING_RATE
            for (i, di) in zip(correct_sequence, dcorrect_inputs):
                self.parameters.embeddings[i] -= embedding_learning_rate * di
            for (i, di) in zip(noise_sequence, dnoise_inputs):
                self.parameters.embeddings[i] -= embedding_learning_rate * di

    def save_word2vec_format(self, fname, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        """
        logger.info("storing %sx%s projection weights into %s" % (self.parameters.vocab_size, self.parameters.embedding_size, fname))

        with open(fname, 'wb') as fout:
            fout.write("%s %s\n" % self.parameters.embeddings.shape)
            # store in sorted order: most frequent words at the top
            for word, count in sorted(vocabulary.words, key=lambda item: -item[1]):
                # word = utils.to_utf8(word)  # always store in utf8
                index = vocabulary.id(word)
                row = self.parameters.embeddings[index]
                if binary:
                    fout.write("%s %s\n" % (word, row.tostring()))
                else:
                    fout.write("%s %s\n" % (word, ' '.join("%f" % val for val in row)))
