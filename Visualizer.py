import os
import glob
import sys
import numpy as np
import trimesh
from keras import Model, Input
from trimesh.viewer import SceneViewer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Model/Encoder'))
sys.path.append(os.path.join(BASE_DIR, 'Model/Decoder'))

from FP_FC import PointNet2Decoder
from PointNet2_SA import PointNet2Encoder

NB_OF_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001

class PointNet2AE(Model):
  def __init__(self, input_shape, latent_dim):
    super(PointNet2AE, self).__init__()
    self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
    self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)
    #self.decoder_model = FC(input_shape[1])

  def call(self, inputs):
    encoded_tensor = self.encoder_model(inputs)
    decoded_tensor = self.decoder_model(encoded_tensor)
    return decoded_tensor

class MySceneViewer(SceneViewer):
    def __init__(self, *args, **kwargs):

        self.scene = trimesh.Scene(trimesh.PointCloud(np.zeros((1, 3))))

        # Load all point clouds from directory
        point_cloud_files = glob.glob(os.path.join("Data/PointClouds"+str(NB_OF_POINTS), "*.npy"))
        point_clouds = [np.load(pc_file) for pc_file in point_cloud_files]
        self.point_clouds = np.stack(point_clouds)

        # Assuming all point clouds have the same shape
        input_shape = point_clouds[0].shape
        print(f'visualizer input_shape: {input_shape}')

        self.autoencoder_model = PointNet2AE(input_shape, 128)
        self.autoencoder_model.build(input_shape=input_shape)

        # # Encoder
        # print(f'point_clouds[0].shape: {input_shape}')
        # self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
        #
        # # Decoder
        # print(f'FC input_shape: {input_shape}')
        # self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)
        # # self.decoder_model = FC(input_shape[1])
        #
        # # Create the autoencoder model
        # input_point_cloud = Input(shape=input_shape)
        # encoded = self.encoder_model(input_point_cloud)
        # decoded = self.decoder_model(encoded)
        # self.autoencoder_model = Model(inputs=input_point_cloud, outputs=decoded)

        # Load model weights
        weight_file = 'Weights/PN2_SA_FP_EMD_' + str(EPOCHS) + 'EP_' + '{:.0e}'.format(LEARNING_RATE) + 'LR_' + str(BATCH_SIZE) + 'BS_' + str(NB_OF_POINTS) + 'PT.h5'
        self.autoencoder_model.load_weights(weight_file)

        self.display_scene()

        super().__init__(self.scene, *args, **kwargs)


    def display_scene(self):
        list = [2330, 1700, 5184, 4427, 3746, 1789, 2342, 3360, 4238, 3428, 621, 4620]
        x_margin = 0
        sceneDict = {}
        for id in list:
            point_cloud = self.point_clouds[id]

            # Predict the selected point cloud
            decoded_point_cloud = self.autoencoder_model.predict(np.expand_dims(point_cloud, axis=0))

            original_cloud = trimesh.points.PointCloud(point_cloud + [x_margin, 0, 0])
            decoded_cloud = trimesh.points.PointCloud(decoded_point_cloud[0] + [x_margin + 1.5, 0, 0])
            #print(decoded_point_cloud)

            sceneDict.update({f"Original {x_margin}": original_cloud, f"Decoded {x_margin}": decoded_cloud})
            x_margin += 5

        self.scene = trimesh.Scene(sceneDict)

if __name__ == "__main__":
    viewer = MySceneViewer()

    #        self.autoencoder_model = Model(inputs=encoded_tensor, outputs=decoded_tensor)
