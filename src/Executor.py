import sys
from keras.layers import Dense, Convolution2D, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from Vgg16 import Vgg16
from IPython.display import FileLink

from utils import onehot
from utils import save_array
from utils import load_array


class Executor:

    def __init(self):
        self.runID = None

        self.vgg = None
        self.batch_size = None
        self.train_batches = None
        self.val_batches = None
        self.data_path = None
        self.learn_rate = None

        self.val_precomputed = None
        self.trn_precomputed = None
        self.val_classes = None
        self.trn_classes = None
        self.val_labels = None
        self.trn_labels = None

        self.conv_layers = None
        self.rescaled_fc_model = None
        self.num_softmax_classes = None
        self.dropout = None

    def and_(self):
        return self;

    def set_Vgg(self, vgg):
        self.vgg = vgg
        self.dropout = 0.5

    def init_validation_and_training_data(self):
        train_path = self.data_path+"train"
        val_path = self.data_path+"valid"

        self.train_batches = self.vgg.get_batches(batch_size=self.batch_size, path=train_path)
        self.val_batches = self.vgg.get_batches(batch_size=self.batch_size, path=val_path)

        print("initialized training data from: "+train_path)
        print("initialized validation data from: "+val_path)

        self.num_softmax_classes = self.train_batches.nb_class
        print("found number of softmax classes: "+self.num_softmax_classes)

    def replace_and_tune_softmax_layer_for_epochs(self, num_epochs=4):
        '''
        In this method, the last dense layer from VGG will be replaced with a
        custom softmax layer and (only) it will be trained for specified epochs.
        The other dense layers will be made non-trainable (happens in the finetune
        method)
        :param num_epochs: Specify num_epochs to finetune only the softmax layer
        :return:
        '''
        self.vgg.finetune(self.train_batches)

        self.vgg.fit_generator(self.train_batches, self.val_batches, nb_epoch=num_epochs)
        print("Vgg model finetuned.")
        return self;

    def save_model_to_file(self):

        fileName = "weights." + self.runID + ".h5"
        self.vgg.model.save_weights(fileName)
        return self;

    def load_model_from_file(self):

        fileName = "weights." + self.runID + ".h5"
        self.vgg.model.load_weights(fileName)
        return self;

    def build_predictions_on_test_data(self):
        test_path = self.data_path + "test"
        b, p= self.vgg.test(test_path, batch_size=2)

        self.prediction = zip([name[8:] for name in b.filenames], p.astype('str'))
        return self;

    def save_predictions_to_file(self):
        fileName = "predictions." + self.runID + ".h5"

        outF = open(fileName, 'w')
        outF.write('image_name,Type_1,Type_2,Type_3\n')

        for elem in self.prediction:
            outF.write(elem[0] + ',' + ','.join(elem[1]) + '\n')
        outF.close()
        FileLink(fileName)
        return self;

    def make_linear_layers_trainable(self):

        layers = self.vgg.model.layers
        # Get the index of the first dense layer...
        first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
        print("first dense layer at index: ", first_dense_idx)

        # ...and set this and all subsequent layers to trainable
        for layer in layers[first_dense_idx:]: layer.trainable=True
        print("all dense layers set trainable.")

        return self

    def compile(self, learn_rate=0.001):
        self.vgg.compile(lr=learn_rate)
        return self

    def init_conv_and_fc_model(self):
        layers = self.vgg.model.layers
        last_conv_idx = [index for index, layer in enumerate(layers)
                         if type(layer) is Convolution2D][-1]
        self.conv_layers = layers[:last_conv_idx+1]

        self.conv_model = Sequential(self.conv_layers)
        self.fc_layers = layers[last_conv_idx+1:]
        return self

    def precompute_conv_model_outputs(self):

        self.init_conv_and_fc_models()

        self.val_classes = self.val_batches.classes
        self.val_labels = onehot(self.val_classes)

        self.trn_classes = self.trn_classes
        self.trn_labels = onehot(self.trn_labels)

        self.val_precomputed = self.conv_model.predict_generator(self.val_batches, self.val_batches.nb_sample)
        self.trn_precomputed = self.conv_model.predict_generator(self.trn_batches, self.trn_batches.nb_sample)

        self.save_precomputed_conv_models()
        return self

    def save_precomputed_conv_models(self):
        fName1 = "precomputed_trn_features."+self.runID+".h5"
        fName2 = "precomputed_val_features."+self.runID+".h5"

        save_array(fName1, self.trn_precomputed)
        save_array(fName2, self.val_precomputed)

        return self;

    def load_precomputed_conv_models(self):
        fName1 = "precomputed_trn_features."+self.runID+".h5"
        fName2 = "precomputed_val_features."+self.runID+".h5"

        load_array(fName1, self.trn_precomputed)
        load_array(fName2, self.val_precomputed)

        return self;

    def proc_wgts(self, layer, prev_dropout, new_dropout):
        '''
        evaluates an array of rescaled weights

        :param layer: fc layer
        :param prev_dropout: prev dropout with which the layer was trained
        :param new_dropout:  new dropout to apply
        :return: array of rescaled weights
        '''
        scale = (1.0 - prev_dropout)/(1.0 - new_dropout)
        return [o * scale for o in layer.get_weights]

    def get_rescaled_fc_model(self, new_dropout):
        '''
        prerequisite:
        fc_layers should be a finetuned model from the previous dropout value.
        Hence, ensure that self.vgg has been fully trained/tuned to the dataset.

        :param new_dropout: new dropout probability to keep.
        :return: new fc model whose weights in the fc layers have been rescaled
        '''
        opt = RMSprop(lr=0.00001, rho=0.7)

        model = Sequential([
            MaxPooling2D(input_shape=self.conv_layers[-1].output_shape[1:]),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(new_dropout),
            Dense(4096, activation='relu'),
            Dropout(new_dropout),
            Dense(self.num_softmax_classes, activation='softmax')
        ])

        for l1, l2 in zip(model.layers, self.fc_layers):
            l1.set_weights(self.proc_wgts(l2, self.dropout, new_dropout))

        # update dropout
        self.dropout = new_dropout
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def init_and_fit_rescaled_fc_model(self, new_dropout):
        '''
        prerequisites:
        the conv_model should've been used to precompute outputs for both the trn
        and val datasets.

        :param new_dropout: new dropout to apply
        :return: a finely tuned fc model whose weights in the fc layers have been rescaled.
        '''

        self.rescaled_fc_model = self.get_rescaled_fc_model(new_dropout)

        # such a finely tuned model needs to be updated very slowly...
        self.rescaled_fc_model.fit(self.trn_precomputed, self.trn_labels, nb_epoch=8,
                       batch_size= self.batch_size, validation_data=(self.val_precomputed, self.val_labels))

        self.rescaled_fc_model.save_weights("rescaled_and_tuned_fc_model."+self.runID+".dropout."+str(new_dropout)+".h5")

class ExecutorBuilder:

    def __init__(self):
        self.executor = Executor()
        self.train_dense_layers = False

    def and_(self):
        return self

    def with_Vgg16(self):
        self.vgg = Vgg16()
        print("Pretrained Vgg16 model loaded.")
        return self

    def data_on_path(self, data_folder):
        self.executor.data_path = data_folder
        return self

    def train_batch_size(self, batch_size):
        self.executor.batch_size = batch_size
        return self

    def learn_rate(self, val):
        self.executor.learn_rate = val
        return self

    def trainable_linear_layers(self, condition=True):
        self.train_dense_layers = condition
        return self

    def build(self):
        self.executor.set_Vgg(self.vgg)
        self.executor.init_validation_and_training_data()

        if(self.train_dense_layers):
            self.executor.make_linear_layers_trainable()

        #TODO: Figure out why this throws an error!!
        #      self.executor.compile(self.learn_rate)
        return self.executor



if __name__ == "__main__":
    # get parameters from system arguments
    train_epochs = int(sys.argv[1])
    learn_rate = float(sys.argv[2])
    data_path = sys.argv[3]

    print("train_epochs: ", train_epochs)
    print("learn_rate", learn_rate)
    print ("data_path: ", data_path)

    executor = ExecutorBuilder().\
        with_Vgg16().\
        and_().\
        train_batch_size(2). \
        and_(). \
        learn_rate(learn_rate).\
        and_().\
        data_on_path(data_path).\
        and_().\
        trainable_linear_layers().\
        build()

    executor.replace_and_tune_softmax_layer_for_epochs(train_epochs).and_().save_model_to_file().and_().\
        build_predictions_on_test_data().and_().save_predictions_to_file()

    executor.precompute_conv_model_outputs();