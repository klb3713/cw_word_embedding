# -*- coding: utf-8 -*-
__author__ = 'klb3713'

"""
Save and load training state.
"""

import os
import time
import config


def creat_run_dir():
    """
    For these variables, create a canonical directory to hold this run's output.
    Write the module values to 'LOGS/run_{date}/'
    """

    run_dir = config.DATA_DIR + "LOGS/run_%s" % time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    return run_dir
