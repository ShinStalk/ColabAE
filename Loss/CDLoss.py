import tensorflow as tf
from keras.losses import Loss

class ChamferDistanceLoss(Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        dist1, dist2 = self.nn_distance_cpu(y_pred, y_true)
        loss = tf.reduce_mean(dist1 + dist2)
        return loss

    def nn_distance_cpu(self, pc1, pc2):
        N = pc1.get_shape()[1]
        M = pc2.get_shape()[1]

        pc1_expand_tile = tf.tile(tf.expand_dims(pc1,2), [1,1,M,1])
        pc2_expand_tile = tf.tile(tf.expand_dims(pc2,1), [1,N,1,1])

        pc_diff = pc1_expand_tile - pc2_expand_tile # B,N,M,C
        pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1) # B,N,M

        dist1 = tf.reduce_min(pc_dist, axis=2) # B,N
        dist2 = tf.reduce_min(pc_dist, axis=1) # B,M

        return dist1, dist2
