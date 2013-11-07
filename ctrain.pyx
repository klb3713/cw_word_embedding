# -*- coding: utf-8 -*-
__author__ = 'klb3713'

import numpy
cimport numpy
import config

DTYPE = numpy.float64
ctypedef numpy.float64_t DTYPE_t

def train_sample_pair(self, numpy.ndarray[DTYPE_t, ndim=2] correct_inputs, numpy.ndarray[DTYPE_t, ndim=2] noise_inputs,
                      double learning_rate):
    cdef numpy.ndarray[DTYPE_t, ndim=2] hidden_weights = self.parameters.hidden_weights
    cdef numpy.ndarray[DTYPE_t, ndim=2] hidden_biases = self.parameters.hidden_biases
    cdef numpy.ndarray[DTYPE_t, ndim=2] output_weights = self.parameters.output_weights
    cdef numpy.ndarray[DTYPE_t, ndim=2] output_biases = self.parameters.output_biases

    cdef numpy.ndarray[DTYPE_t, ndim=2] correct_prehidden = numpy.dot(correct_inputs, hidden_weights) + hidden_biases
    correct_hiddenl = []
    for x in correct_prehidden[0]:
        if x < -1:
            correct_hiddenl.append(-1.)
        elif x > 1:
            correct_hiddenl.append(1.)
        else:
            correct_hiddenl.append(x)
    cdef numpy.ndarray[DTYPE_t, ndim=2] correct_hidden = numpy.asarray(correct_hiddenl).reshape((1, self.parameters.hidden_size))
    cdef numpy.ndarray[DTYPE_t, ndim=2] correct_score = numpy.dot(correct_hidden, output_weights) + output_biases

    cdef numpy.ndarray[DTYPE_t, ndim=2] noise_prehidden = numpy.dot(noise_inputs, hidden_weights) + hidden_biases
    noise_hiddenl = []
    for x in noise_prehidden[0]:
        if x < -1:
            noise_hiddenl.append(-1.)
        elif x > 1:
            noise_hiddenl.append(1.)
        else:
            noise_hiddenl.append(x)
    cdef numpy.ndarray[DTYPE_t, ndim=2] noise_hidden = numpy.asarray(noise_hiddenl).reshape((1, self.parameters.hidden_size))
    cdef numpy.ndarray[DTYPE_t, ndim=2] noise_score = numpy.dot(noise_hidden, output_weights) + output_biases

    cdef numpy.ndarray[DTYPE_t, ndim=2] loss = 1 - correct_score + noise_score
    if loss < 0:
        return 0

    # gradients for correct sample
    cdef int c_dcost_dout = -1
    cdef numpy.ndarray[DTYPE_t, ndim=2] c_doutput_weights = numpy.dot(correct_hidden.T, c_dcost_dout)
    cdef int c_doutput_biases = c_dcost_dout
    cdef numpy.ndarray[DTYPE_t, ndim=2] c_dcost_dhidden = numpy.dot(c_dcost_dout, output_weights.T)

    c_dcost_dinputl = []
    # c_dcost_dhidden_v = c_dcost_dhidden.reshape((self.parameters.hidden_size,))
    cdef int i
    for i, x in enumerate(correct_prehidden[0]):
        if x < -1 or x > 1:
            c_dcost_dinputl.append(0.)
        else:
            c_dcost_dinputl.append(c_dcost_dhidden[0, i])
    cdef numpy.ndarray[DTYPE_t, ndim=2] c_dcost_dinput = numpy.asarray(c_dcost_dinputl).reshape((1, self.parameters.hidden_size))

    cdef numpy.ndarray[DTYPE_t, ndim=2] c_dhidden_weights = numpy.dot(correct_inputs.T, c_dcost_dinput)
    cdef numpy.ndarray[DTYPE_t, ndim=2] c_dhidden_biases = c_dcost_dinput
    cdef numpy.ndarray[DTYPE_t, ndim=2] c_dcost_lt = numpy.dot(c_dcost_dinput, hidden_weights.T)

    c_dcost_lt = c_dcost_lt.reshape(self.parameters.window_size, self.parameters.embedding_size)
    cdef numpy.ndarray[DTYPE_t, ndim=2] dcorrect_input = c_dcost_lt.sum(axis=0) - c_dcost_lt

    # gradients for noise sample
    cdef int n_dcost_dout = 1
    cdef numpy.ndarray[DTYPE_t, ndim=2] n_doutput_weights = numpy.dot(noise_hidden.T, n_dcost_dout)
    cdef int n_doutput_biases = n_dcost_dout
    cdef numpy.ndarray[DTYPE_t, ndim=2] n_dcost_dhidden = numpy.dot(n_dcost_dout, output_weights.T)

    n_dcost_dinputl = []
    # n_dcost_dhidden_v = n_dcost_dhidden.reshape((self.parameters.hidden_size,))
    for i, x in enumerate(noise_prehidden[0]):
        if x < -1 or x > 1:
            n_dcost_dinputl.append(0.)
        else:
            n_dcost_dinputl.append(n_dcost_dhidden[0, i])
    cdef numpy.ndarray[DTYPE_t, ndim=2] n_dcost_dinput = numpy.asarray(n_dcost_dinputl).reshape((1, self.parameters.hidden_size))

    cdef numpy.ndarray[DTYPE_t, ndim=2] n_dhidden_weights = numpy.dot(noise_inputs.T, n_dcost_dinput)
    cdef numpy.ndarray[DTYPE_t, ndim=2] n_dhidden_biases = n_dcost_dinput
    cdef numpy.ndarray[DTYPE_t, ndim=2] n_dcost_lt = numpy.dot(n_dcost_dinput, hidden_weights.T)

    n_dcost_lt = n_dcost_lt.reshape(self.parameters.window_size, self.parameters.embedding_size)
    cdef numpy.ndarray[DTYPE_t, ndim=2] dnoise_input = n_dcost_lt.sum(axis=0) - n_dcost_lt

    # update parameters
    self.parameters.hidden_weights -= learning_rate * (c_dhidden_weights + n_dhidden_weights)
    self.parameters.hidden_biases -= learning_rate * (c_dhidden_biases + n_dhidden_biases)
    self.parameters.output_weights -= learning_rate * (c_doutput_weights + n_doutput_weights)
    self.parameters.output_biases -= learning_rate * (c_doutput_biases + n_doutput_biases)

    return dcorrect_input, dnoise_input, loss, correct_score, noise_score

def train(self, correct_sequences):
    cdef double learning_rate = config.LEARNING_RATE
    cdef int i
    cdef numpy.ndarray[DTYPE_t, ndim=2] emb_correct
    cdef numpy.ndarray[DTYPE_t, ndim=2] emb_noise

    cdef numpy.ndarray[DTYPE_t, ndim=1] di
    cdef numpy.ndarray[DTYPE_t, ndim=2] dcorrect_input
    cdef numpy.ndarray[DTYPE_t, ndim=2] dnoise_input
    cdef numpy.ndarray[DTYPE_t, ndim=2] loss
    cdef numpy.ndarray[DTYPE_t, ndim=2] correct_score
    cdef numpy.ndarray[DTYPE_t, ndim=2] noise_score

    for correct_sequence in correct_sequences:
        emb_correct = self.embed(correct_sequence)
        for noise_sequence in self.corrupt_examples(correct_sequence):
            emb_noise = self.embed(noise_sequence)
            r = train_sample_pair(self, emb_correct, emb_noise, learning_rate)
            if r == 0:
                continue
            (dcorrect_input, dnoise_input, loss, correct_score, noise_score) = r

            self.train_loss += loss
            self.train_err += 1 if correct_score <= noise_score else 0
            self.train_lossnonzero += 1 if loss > 0 else 0

            # embedding_learning_rate = config.EMBEDDING_LEARNING_RATE * weights[0]
            for (i, di) in zip(correct_sequence, dcorrect_input):
                self.parameters.embeddings[i] -= learning_rate * di

            for (i, di) in zip(noise_sequence, dnoise_input):
                self.parameters.embeddings[i] -= learning_rate * di
