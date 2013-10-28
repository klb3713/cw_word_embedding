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
from movingaverage import MovingAverage


class Model(object):
    """
    A Model can:

    @type parameters: L{Parameters}
    @todo: Document
    """

    def __init__(self):
        vocabulary.load_vocabulary()
        self.parameters = Parameters()
        self.train_loss = MovingAverage()
        self.train_err = MovingAverage()
        self.train_lossnonzero = MovingAverage()
        self.train_squash_loss = MovingAverage()
        self.train_unpenalized_loss = MovingAverage()
        self.train_l1penalty = MovingAverage()
        self.train_unpenalized_lossnonzero = MovingAverage()
        self.train_correct_score = MovingAverage()
        self.train_noise_score = MovingAverage()
        self.train_cnt = 0
        self.COMPILE_MODE = theano.compile.Mode('c|py', 'fast_run')

        self.train_function = self._get_train_function()

    def __getstate__(self):
        return (self.parameters,
                self.train_loss,
                self.train_err,
                self.train_lossnonzero,
                self.train_squash_loss,
                self.train_unpenalized_loss,
                self.train_l1penalty,
                self.train_unpenalized_lossnonzero,
                self.train_correct_score,
                self.train_noise_score,
                self.train_cnt)

    def __setstate__(self, state):
        (self.parameters,
         self.train_loss,
         self.train_err,
         self.train_lossnonzero,
         self.train_squash_loss,
         self.train_unpenalized_loss,
         self.train_l1penalty,
         self.train_unpenalized_lossnonzero,
         self.train_correct_score,
         self.train_noise_score,
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
        (dhidden_weights,
         dhidden_biases,
         doutput_weights,
         doutput_biases) = T.grad(total_loss,
                                  [self.parameters.hidden_weights,
                                   self.parameters.hidden_biases,
                                   self.parameters.output_weights,
                                   self.parameters.output_biases])

        dcorrect_inputs = T.grad(total_loss, correct_inputs)
        dnoise_inputs = T.grad(total_loss, noise_inputs)

        para_gpara = zip((self.parameters.hidden_weights,
                          self.parameters.hidden_biases,
                          self.parameters.output_weights,
                          self.parameters.output_biases),
                         (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases))
        updates = [(p, p - learning_rate * gp) for p, gp in para_gpara]

        print("About to compile train function...")
        train_function = theano.function([correct_inputs, noise_inputs, learning_rate],
                                         [dcorrect_inputs, dnoise_inputs, unpenalized_loss, correct_score, noise_score],
                                         mode=self.COMPILE_MODE,
                                         updates=updates)
        print("Done constructing function for train")
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
        return numpy.array(embs)

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
        (dcorrect_inputss, dnoise_inputss, losses, correct_scores, noise_scores) = r

        to_normalize = []
        for index in range(len(correct_sequences)):
            (loss, correct_score, noise_score) = (losses[index], correct_scores[index], noise_scores[index])

            correct_sequence = correct_sequences[index]
            noise_sequence = noise_sequences[index]

            dcorrect_inputs = dcorrect_inputss[index].reshape((config.WINDOW_SIZE, config.EMBEDDING_SIZE))
            dnoise_inputs = dnoise_inputss[index].reshape((config.WINDOW_SIZE, config.EMBEDDING_SIZE))

            self.train_loss.add(loss)
            self.train_err.add(correct_score <= noise_score)
            self.train_lossnonzero.add(loss > 0)
            squash_loss = 1./(1.+math.exp(-loss))
            self.train_squash_loss.add(squash_loss)
            self.train_correct_score.add(correct_score)
            self.train_noise_score.add(noise_score)
    
            self.train_cnt += 1
            if self.train_cnt % 100000 == 0:
                logging.info(("After %d updates, pre-update train loss %s" % (self.train_cnt, self.train_loss.verbose_string())))
                logging.info(("After %d updates, pre-update train error %s" % (self.train_cnt, self.train_err.verbose_string())))
                logging.info(("After %d updates, pre-update train Pr(loss != 0) %s" % (self.train_cnt, self.train_lossnonzero.verbose_string())))
                logging.info(("After %d updates, pre-update train squash(loss) %s" % (self.train_cnt, self.train_squash_loss.verbose_string())))
                logging.info(("After %d updates, pre-update train correct score %s" % (self.train_cnt, self.train_correct_score.verbose_string())))
                logging.info(("After %d updates, pre-update train noise score %s" % (self.train_cnt, self.train_noise_score.verbose_string())))


            embedding_learning_rate = config.EMBEDDING_LEARNING_RATE * weights[0]
            if loss == 0:
                for di in dcorrect_inputs + dnoise_inputs:
                    assert (di == 0).all()
            else:
                for (i, di) in zip(correct_sequence, dcorrect_inputs):
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if config.NORMALIZE_EMBEDDINGS:
                        to_normalize.add(i)
                for (i, di) in zip(noise_sequence, dnoise_inputs):
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if config.NORMALIZE_EMBEDDINGS:
                        to_normalize.add(i)

        if len(to_normalize) > 0:
            self.parameters.normalize(to_normalize)
