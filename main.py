# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import config
import samples
import state
import model_gpu
# import cProfile, pstats, StringIO

logger = logging.getLogger(__name__)


def train(debug=False):
    run_dir = state.creat_run_dir()
    # logfile = os.path.join(run_dir, "log")
    # logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info("INITIALIZING...")
    cw_model = model_gpu.Model()
    cnt = 0
    epoch = 1
    train_mini_batchs = samples.TrainingMiniBatchStream()
    logger.info("INITIALIZING TRAINING STATE")

    while epoch <= config.EPOCH:
        logger.info("STARTING TRAINING...")
        logger.info("STARTING EPOCH #%d" % epoch)

        # pr = cProfile.Profile()
        # pr.enable()

        for batch in train_mini_batchs:
            cnt += len(batch)
            cw_model.train(batch)

            if debug and cnt % (int(100000./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                logger.info("FINISH TRAINED %d SAMPLES of epoch #%d." % (cnt, epoch))

                # pr.disable()
                # s = StringIO.StringIO()
                # sortby = 'cumulative'
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()
                # print s.getvalue()
                # exit()

        # save embedding for every epoch
        cw_model.save_word2vec_format(os.path.join(run_dir, config.VECTOR_FILE + '_epoch%d.bin' % epoch), binary=True)
        cw_model.save_word2vec_format(os.path.join(run_dir, config.VECTOR_FILE + '_epoch%d.txt' % epoch), binary=False)

        logger.info("After #%d epoch updates, train loss: %f" % (epoch, cw_model.train_loss))
        logger.info("After #%d epoch updates, train error: %d" % (epoch, cw_model.train_err))
        logger.info("After #%d epoch updates, train loss nonzero: %d" % (epoch, cw_model.train_lossnonzero))
        cw_model.reset()
        logger.info("FINISH TRAIN EPOCH #%d" % epoch)

        train_mini_batchs = samples.TrainingMiniBatchStream()
        epoch += 1


if __name__ == "__main__":
    train(debug=True)