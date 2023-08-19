import os, sys
import tensorflow as tf
from keras.losses import Loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'Ops/approxmatch'))

import approxmatch

class EMDLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        match = approxmatch.approx_match(y_true, y_pred)
        loss = tf.reduce_mean(approxmatch.match_cost(y_true, y_pred, match))

        return loss