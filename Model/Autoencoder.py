import os
import sys
from keras import Model, Input
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Encoder'))
sys.path.append(os.path.join(BASE_DIR, 'Decoder'))
sys.path.append(os.path.join(BASE_DIR, '../Loss'))

from PointNet import PointNet
from FC import FC
from CDLoss import ChamferDistanceLoss

class PointNetAutoencoder:
    def __init__(self, input):
        self.input = input
        self.model = None

    def compile(self, learning_rate=0.0001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=ChamferDistanceLoss())

    def build(self):
        encoder_model = PointNet(self.input).build()
        encoded_tensor = encoder_model(self.input)

        decoder_model = FC(encoded_tensor).build()
        decoded_tensor = decoder_model(encoded_tensor)

        self.model = Model(inputs=self.input, outputs=decoded_tensor)

        return self.model

if __name__ == "__main__":
    input_tensor = Input(shape=(1024, 3, 1))
    ae = PointNetAutoencoder(input_tensor)
    ae.build()
    ae.compile()