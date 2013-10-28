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


def train():
    run_dir = state.creat_run_dir()
    logfile = os.path.join(run_dir, "log")

    print("INITIALIZING...")
    cw_model = model.Model()
    cnt = 0
    epoch = 1
    train_mini_batchs = samples.TrainingMiniBatchStream()
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("INITIALIZING TRAINING STATE")

    while epoch <= config.EPOCH:
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        logging.info("%s: STARTING TRAINING..." % cur_time)
        logging.info("%s: STARTING EPOCH #%d" % (cur_time, epoch))
        for batch in train_mini_batchs:
            cnt += len(batch)
            cw_model.train(batch)

            if cnt % (int(1000./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                logging.info("Finished training step %d (epoch %d)" % (cnt, epoch))

            if cnt % (int(100000./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                logging.info("%s: EXIT TRAINED 100000 SAMPLES." % cur_time)
                sys.exit(0)

        train_mini_batchs = samples.TrainingMiniBatchStream()
        epoch += 1
        # todo save embedding for every epoch
        # cw_model.save()


if __name__ == "__main__":
    train()