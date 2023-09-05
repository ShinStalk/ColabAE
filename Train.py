import glob
import os
import sys

import numpy as np

NB_OF_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Loading the point clouds data
point_cloud_files = glob.glob(os.path.join("Data/PointClouds"+str(NB_OF_POINTS), "*.npy"))
point_clouds = [np.load(pc_file) for pc_file in point_cloud_files]
point_clouds = np.stack(point_clouds)

import tensorflow as tf
from keras import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'Loss'))
sys.path.append(os.path.join(BASE_DIR, 'Model/Decoder'))
sys.path.append(os.path.join(BASE_DIR, 'Model/Encoder'))

from PN2_MSG import PointNet2Encoder
from FC_UPCONV import PointNet2Decoder

from tensorflow.keras.optimizers import Adam
#from EMDLoss import EMDLoss
from CDLoss import ChamferDistanceLoss

class PointNet2AE(Model):
  def __init__(self, input_shape, latent_dim):
    super(PointNet2AE, self).__init__()
    self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
    self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)
    #self.decoder_model = FC(input_shape[1])

  def call(self, inputs):
    xyz, points = self.encoder_model(inputs)
    decoded_tensor = self.decoder_model(xyz, points)
    return decoded_tensor


if __name__ == "__main__":
  ae_model = PointNet2AE(point_clouds.shape[1:], 128)
  ae_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=ChamferDistanceLoss())

  # Train the model
  history = ae_model.fit(point_clouds, point_clouds, epochs=EPOCHS, batch_size=BATCH_SIZE)

  # plot training history
  import matplotlib.pyplot as plt

  plt.plot(history.history['loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train'], loc='upper right')

  file_name = 'PN2_SA_FP_EMD_' + str(EPOCHS) + 'EP_' + '{:.0e}'.format(LEARNING_RATE) + 'LR_' + str(BATCH_SIZE) + 'BS_' + str(NB_OF_POINTS) + 'PT.h5'
  print(f'file_name: {file_name}')
  ae_model.save_weights(os.path.join('Weights/', file_name))
