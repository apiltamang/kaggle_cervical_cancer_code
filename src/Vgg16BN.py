from Vgg16 import Vgg16
from keras.layers import Dense, BatchNormalization, Dropout, Lambda, Flatten
from keras.models import Sequential
from keras.utils.data_utils import get_file
import os

class Vgg16BN(Vgg16):
    """The VGG 16 Imagenet model with Batch
        Normalization for the Dense Layers
    """

    def __init__(self):
        models_dir = os.getcwd()+"/models/"
        self.FILE_PATH = 'file://'+models_dir
        self.create(size=(224,224), include_top=True)
        super(Vgg16BN, self).get_classes()

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    def create(self, size, include_top):
        if size != (224,224):
            include_top=False

        model = self.model = Sequential()
        model.add(Lambda(self.vgg_preprocess, input_shape=(3,)+size, output_shape=(3,)+size))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        if not include_top:
            fname = 'vgg16_bn_conv.h5'
            model.load_weights(super(Vgg16BN, self).get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
            return

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16_bn.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


