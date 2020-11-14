from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

class SRCNN:

    @staticmethod
    def build(height, width, channels):

        # Input shape: 33x33x3
        input_shape = (height, width, channels)
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)

        model = Sequential()
        
        # Output shape: 25x25x64
        model.add(Conv2D(64, (9, 9), kernel_initializer="he_normal", input_shape=input_shape))
        model.add(Activation("relu"))

        # Output shape: 25x25x32
        model.add(Conv2D(32, (1, 1), kernel_initializer="he_normal"))
        model.add(Activation("relu"))

        # Output shape 21x21xchannels
        model.add(Conv2D(channels, (5, 5), kernel_initializer="he_normal"))
        model.add(Activation("relu"))

        return model
