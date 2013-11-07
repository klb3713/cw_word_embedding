# -*- coding: utf-8 -*-
__author__ = 'klb3713'

import ctrain
import logging
import copy
import numpy
import config
import vocabulary
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

    def train_sample_pair(self, correct_inputs, noise_inputs, learning_rate):
        correct_prehidden = numpy.dot(correct_inputs, self.parameters.hidden_weights) + self.parameters.hidden_biases
        correct_hidden = []
        for x in correct_prehidden[0]:
            if x < -1:
                correct_hidden.append(-1.)
            elif x > 1:
                correct_hidden.append(1.)
            else:
                correct_hidden.append(x)
        correct_hidden = numpy.asarray(correct_hidden).reshape((1, self.parameters.hidden_size))
        correct_score = numpy.dot(correct_hidden, self.parameters.output_weights) + self.parameters.output_biases

        noise_prehidden = numpy.dot(noise_inputs, self.parameters.hidden_weights) + self.parameters.hidden_biases
        noise_hidden = []
        for x in noise_prehidden[0]:
            if x < -1:
                noise_hidden.append(-1.)
            elif x > 1:
                noise_hidden.append(1.)
            else:
                noise_hidden.append(x)
        noise_hidden = numpy.asarray(noise_hidden).reshape((1, self.parameters.hidden_size))
        noise_score = numpy.dot(noise_hidden, self.parameters.output_weights) + self.parameters.output_biases

        loss = 1 - correct_score + noise_score
        if loss < 0:
            return 0

        # gradients for correct sample
        c_dcost_dout = -1
        c_doutput_weights = numpy.dot(correct_hidden.T, c_dcost_dout)
        c_doutput_biases = c_dcost_dout
        c_dcost_dhidden = numpy.dot(c_dcost_dout, self.parameters.output_weights.T)

        c_dcost_dinput = []
        c_dcost_dhidden_v = c_dcost_dhidden.reshape((self.parameters.hidden_size,))
        for i, x in enumerate(correct_prehidden[0]):
            if x < -1 or x > 1:
                c_dcost_dinput.append(0.)
            else:
                c_dcost_dinput.append(c_dcost_dhidden_v[i])
        c_dcost_dinput = numpy.asarray(c_dcost_dinput).reshape((1, self.parameters.hidden_size))

        c_dhidden_weights = numpy.dot(correct_inputs.T, c_dcost_dinput)
        c_dhidden_biases = c_dcost_dinput
        c_dcost_lt = numpy.dot(c_dcost_dinput, self.parameters.hidden_weights.T)

        c_dcost_lt = c_dcost_lt.reshape(self.parameters.window_size, self.parameters.embedding_size)
        dcorrect_input = c_dcost_lt.sum(axis=0) - c_dcost_lt

        # gradients for noise sample
        n_dcost_dout = 1
        n_doutput_weights = numpy.dot(noise_hidden.T, n_dcost_dout)
        n_doutput_biases = n_dcost_dout
        n_dcost_dhidden = numpy.dot(n_dcost_dout, self.parameters.output_weights.T)

        n_dcost_dinput = []
        n_dcost_dhidden_v = n_dcost_dhidden.reshape((self.parameters.hidden_size,))
        for i, x in enumerate(noise_prehidden[0]):
            if x < -1 or x > 1:
                n_dcost_dinput.append(0.)
            else:
                n_dcost_dinput.append(n_dcost_dhidden_v[i])
        n_dcost_dinput = numpy.asarray(n_dcost_dinput).reshape((1, self.parameters.hidden_size))

        n_dhidden_weights = numpy.dot(noise_inputs.T, n_dcost_dinput)
        n_dhidden_biases = n_dcost_dinput
        n_dcost_lt = numpy.dot(n_dcost_dinput, self.parameters.hidden_weights.T)

        n_dcost_lt = n_dcost_lt.reshape(self.parameters.window_size, self.parameters.embedding_size)
        dnoise_input = n_dcost_lt.sum(axis=0) - n_dcost_lt

        dhidden_weights = c_dhidden_weights + n_dhidden_weights
        dhidden_biases = c_dhidden_biases + n_dhidden_biases

        doutput_weights = c_doutput_weights + n_doutput_weights
        doutput_biases = c_doutput_biases + n_doutput_biases

        # update parameters
        self.parameters.hidden_weights -= learning_rate * dhidden_weights
        self.parameters.hidden_biases -= learning_rate * dhidden_biases
        self.parameters.output_weights -= learning_rate * doutput_weights
        self.parameters.output_biases -= learning_rate * doutput_biases

        return dcorrect_input, dnoise_input, loss, correct_score, noise_score

    def embed(self, sequence):
        """
        Embed one sequence of vocabulary IDs.
        """
        embs = numpy.concatenate([self.parameters.embeddings[s] for s in sequence]).reshape((1, self.parameters.input_size))
        return embs

    def corrupt_examples(self, correct_sequence):
        noise_sequences = []
        weights = []
        half_window = config.WINDOW_SIZE / 2

        for i in xrange(200):
            noise_sequence = copy.copy(correct_sequence)
            noise_sequence[half_window] = i
            # weight = self.parameters.vocab_size
            noise_sequences.append(noise_sequence)
            # weights.append(weight)

        return noise_sequences

    def train(self, correct_sequences):
        # ctrain.train(self, correct_sequences)

        learning_rate = config.LEARNING_RATE
        for correct_sequence in correct_sequences:
            emb_correct = self.embed(correct_sequence)
            for noise_sequence in self.corrupt_examples(correct_sequence):
                emb_noise = self.embed(noise_sequence)
                # r = ctrain.train_sample_pair(self, emb_correct, emb_noise, learning_rate)
                r = self.train_sample_pair(emb_correct, emb_noise, learning_rate)
                if r == 0:
                    continue
                (dcorrect_input, dnoise_input, loss, correct_score, noise_score) = r

                self.train_loss += loss
                self.train_err += 1 if correct_score <= noise_score else 0
                self.train_lossnonzero += 1 if loss > 0 else 0

                # embedding_learning_rate = config.EMBEDDING_LEARNING_RATE * weights[0]
                embedding_learning_rate = config.LEARNING_RATE
                for (i, di) in zip(correct_sequence, dcorrect_input):
                    self.parameters.embeddings[i] -= embedding_learning_rate * di

                for (i, di) in zip(noise_sequence, dnoise_input):
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
