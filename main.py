# -*- coding: utf-8 -*-
__author__ = 'klb3713'


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import config
import samples
import state
import model


# def validate(cnt):
#     logranks = []
#     logging.info("BEGINNING VALIDATION AT TRAINING STEP %d" % cnt)
#     logging.info(stats())
#     i = 0
#     for (i, ve) in enumerate(samples.get_validation_example()):
#         logranks.append(math.log(m.validate(ve)))
#         if (i+1) % 10 == 0:
#             logging.info("Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f" % (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks))))
#             logging.info(stats())
#     logging.info("FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d" % (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1))
#     logging.info(stats())


def train():
    run_dir = state.creat_run_dir()
    logfile = os.path.join(run_dir, "log")

    print("INITIALIZING...")
    m = model.Model()
    cnt = 0
    epoch = 1
    get_train_minibatch = samples.TrainingMinibatchStream()
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("INITIALIZING TRAINING STATE")

    while 1:
        logging.info("STARTING EPOCH #%d" % epoch)
        for ebatch in get_train_minibatch:
            cnt += len(ebatch)
            m.train(ebatch)

            if cnt % (int(1000./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                logging.info("Finished training step %d (epoch %d)" % (cnt, epoch))
            if cnt % (int(100000./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                #diagnostics.diagnostics(cnt, m)
                if os.path.exists(os.path.join(run_dir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(run_dir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(run_dir, "BAD"))
                    sys.exit(0)
            # if cnt % (int(config.VALIDATE_EVERY*1./config.MINIBATCH_SIZE) * config.MINIBATCH_SIZE) == 0:
                # state.save(m, cnt, epoch, get_train_minibatch, run_dir, newkeystr)
                # diagnostics.visualizedebug(cnt, m, rundir, newkeystr)

        get_train_minibatch = samples.TrainingMinibatchStream()
        epoch += 1


if __name__ == "__main__":
    train()