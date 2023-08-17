import os
import sys
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Encoder'))
sys.path.append(os.path.join(BASE_DIR, 'Decoder'))

from PointNet import PointNet
from FC import FC

class PointNetAutoencoder:
    def __init__(self, input):
        self.input = input
        self.model = None

    def build(self):
        encoder_model = PointNet(self.input).build()
        encoded_tensor = encoder_model(self.input)

        decoder_model = FC(encoded_tensor).build()
        decoded_tensor = decoder_model(encoded_tensor)

        self.model = Model(inputs=self.input, outputs=decoded_tensor)

        return self.model
