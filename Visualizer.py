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

from PN2_MSG import PointNet2Encoder
from FC_UPCONV import PointNet2Decoder

NB_OF_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

class PointNet2AE(Model):
  def __init__(self, input_shape, latent_dim):
    super(PointNet2AE, self).__init__()
    self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
    self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)

  def call(self, inputs):
    xyz, points = self.encoder_model(inputs)
    decoded_tensor = self.decoder_model(xyz, points)
    return decoded_tensor

class MySceneViewer(SceneViewer):
    def __init__(self, *args, **kwargs):

        self.scene = trimesh.Scene(trimesh.PointCloud(np.zeros((1, 3))))

        # Load all point clouds from directory
        point_cloud_files = glob.glob(os.path.join("Data/PointClouds"+str(NB_OF_POINTS), "*.npy"))
        point_clouds = [np.load(pc_file) for pc_file in point_cloud_files]
        self.point_clouds = np.stack(point_clouds)

        self.autoencoder_model = PointNet2AE(self.point_clouds[0].shape, 128)

        self.display_scene()

        super().__init__(self.scene, *args, **kwargs)


    def display_scene(self):
        print(f'display_scene')
        self.autoencoder_model.build((BATCH_SIZE,) + self.point_clouds[0].shape)

        # Load model weights
        weight_file = 'Weights/PN2_MSG_FCUPCONV_EMD_' + str(EPOCHS) + 'EP_' + '{:.0e}'.format(LEARNING_RATE) + 'LR_' + str(BATCH_SIZE) + 'BS_' + str(NB_OF_POINTS) + 'PT.h5'
        self.autoencoder_model.load_weights(weight_file)

        print(f'weights loaded')

        list = [2330, 1700, 5184, 4427, 3746, 1789, 2342, 3360, 4238, 3428, 621, 4620]
        x_margin = 0
        sceneDict = {}
        for id in list:
            point_cloud = self.point_clouds[id]

            print(f'predict point_cloud: {point_cloud.shape}')
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
