from __future__ import division, print_function

import json
import numpy as np
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.preprocessing import image
import os

class Vgg16(object):
    """The VGG 16 Imagenet model"""

    def __init__(self):
        self.FILE_PATH = 'file://'+os.getcwd()+"/models/"
        self.model = None
        self.create(weights_file="vgg16.h5")
        self.get_classes()
        
    def __init__(self, WEIGHTS_FILE):
        self.FILE_PATH = 'file://'+os.getcwd()+"/models/"
        self.model = None
        self.create(weights_file=WEIGHTS_FILE)
        self.get_classes()

    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models_cache')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))


    def create(self, weights_file):
        model = self.model = Sequential()
        model.add(Lambda(self.vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224)))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        model.load_weights(get_file(weights_file, self.FILE_PATH + weights_file, cache_subdir='models_cache'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=16, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        self.ft(batches.nb_class)
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


    def fit(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    def fit_generator(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)

    def vgg_preprocess(self, x):
        vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
        x = x - vgg_mean
        return x[:, ::-1] # reverse axis rgb->bgr
    
    def get_new_fc_model(self, conv_layer, num_softmax_classes, new_dropout=0.5):
        new_model = Sequential([
            MaxPooling2D(input_shape=conv_layer.output_shape[1:]),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(new_dropout),
            Dense(4096, activation='relu'),
            Dropout(new_dropout),
            Dense(num_softmax_classes, activation='softmax')
        ])
        
        return new_model

    def init_vgg_with_retrained_fc_layers(self, fc_layers):
        layers = self.model.layers
        last_conv_idx = [index for index, layer in enumerate(layers)
                         if type(layer) is Convolution2D][-1]

        for l1, l2 in zip(self.model.layers[last_conv_idx+1:], fc_layers):
            l1.set_weights(l2.get_weights())


