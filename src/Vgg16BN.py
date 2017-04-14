from Vgg16 import Vgg16
from keras.layers import Dense, BatchNormalization, Dropout, Lambda, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils.data_utils import get_file
import os

class Vgg16BN(Vgg16):
    """
    The VGG 16 Imagenet model with Batch
    Normalization for the Dense Layers
    """

    def __init__(self):
        super(Vgg16BN, self).__init__('vgg16_bn.h5')

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    
    def get_new_fc_model(self, conv_layer, num_softmax_classes, new_dropout=0.5):
        new_model = Sequential([
            MaxPooling2D(input_shape=conv_layer.output_shape[1:]),
            Flatten(),
            Dense(4096, activation='relu'),            
            BatchNormalization(),
            Dropout(new_dropout),
            Dense(4096, activation='relu'),
            BatchNormalization(),
            Dropout(new_dropout),
            Dense(num_softmax_classes, activation='softmax')
        ])
        
        return new_model