# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import logging
import config
import samples
import state
import model

logger = logging.getLogger(__name__)

def train():
    logger.info("INITIALIZING...")
    cw_model = model.Model()
    cnt = 0
    epoch = 1
    train_mini_batchs = samples.TrainingMiniBatchStream()
    logger.info("INITIALIZING TRAINING STATE")

    while epoch <= config.EPOCH:
        logger.info("STARTING TRAINING...")
        logger.info("STARTING EPOCH #%d" % epoch)
        for batch in train_mini_batchs:
            cnt += len(batch)
            cw_model.train(batch)

            if cnt % (int(100000./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                logger.info("FINISH TRAINED 100000 SAMPLES of epoch #%d." % epoch)

        # save embedding for every epoch
        cw_model.save_word2vec_format(config.VECTOR_FILE + '_epoch%d.bin' % epoch, binary=True)
        cw_model.save_word2vec_format(config.VECTOR_FILE + '_epoch%d.txt' % epoch, binary=False)

        logger.info("FINISH TRAIN EPOCH #%d" % epoch)
        train_mini_batchs = samples.TrainingMiniBatchStream()
        epoch += 1


if __name__ == "__main__":
    run_dir = state.creat_run_dir()
    # logfile = os.path.join(run_dir, "log")
    # logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    train()