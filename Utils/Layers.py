from keras.layers import Conv2D, Conv2DTranspose, Dense, BatchNormalization, Activation

def conv2d(inputs, num_output_channels, kernel_size):
    conv = Conv2D(num_output_channels, kernel_size)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv


def conv2d_transpose(inputs, num_output_channels, kernel_size, stride=(1,1)):
    conv = Conv2DTranspose(num_output_channels, kernel_size, strides=stride)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    return conv


def fully_connected(inputs, num_outputs):
    dense_layer = Dense(num_outputs)(inputs)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Activation('relu')(dense_layer)

    return dense_layer