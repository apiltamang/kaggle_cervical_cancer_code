import sys
from keras.layers import Dense, Convolution2D, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

from Vgg16 import Vgg16
from IPython.display import FileLink

from Vgg16BN import Vgg16BN
from utils import onehot
from utils import save_array
from utils import load_array
from keras.preprocessing import image

import os

class Executor:

    def __init(self):
        self.runID = None

        self.vgg = None
        self.batch_size = None
        self.train_batches = None
        self.val_batches = None
        self.data_path = None
        self.learn_rate = None
        self.train_path = None
        self.val_path = None

        self.val_precomputed = None
        self.train_precomputed = None
        self.val_classes = None
        self.trn_classes = None
        self.val_labels = None
        self.train_labels = None

        self.conv_layers = None
        self.rescaled_fc_model = None
        self.num_softmax_classes = None
        self.dropout = None
        self.use_precomputed_conv_output = False

        cwd = os.getcwd();
        run_dir = os.path.join(cwd, self.runID)

        if not os.path.exists(run_dir):
            os.mkdirs(run_dir)

    def and_(self):
        return self;

    def set_Vgg(self, vgg):
        self.vgg = vgg
        self.dropout = 0.5

    def init_validation_and_training_data(self):
        '''
        Initializes train_batches and val_batches; arguably, the two most
        important properties. Initialize by default when constructing the
        executor object.
        :return: train_batches, val_batches, num_softmax_classes
        '''
        self.train_path = self.data_path+"train"
        self.val_path = self.data_path+"valid"

        self.train_batches = self.vgg.get_batches(batch_size=self.batch_size, path=self.train_path)
        self.val_batches = self.vgg.get_batches(batch_size=self.batch_size, path=self.val_path)

        print("initialized training data from: "+self.train_path)
        print("initialized validation data from: "+self.val_path)

        self.num_softmax_classes = self.train_batches.nb_class
        print("found number of softmax classes: "+str(self.num_softmax_classes))

    def init_conv_and_fc_models(self):
        '''
        Method initializes the convolution and fc models. Wouldn't
        hurt to initialize them by default.
        :return: initilizes self.conv_layers, self.conv_model and self.fc_layers
        '''
        layers = self.vgg.model.layers
        last_conv_idx = [index for index, layer in enumerate(layers)
                         if type(layer) is Convolution2D][-1]
        self.conv_layers = layers[:last_conv_idx+1]

        self.conv_model = Sequential(self.conv_layers)
        self.fc_layers = layers[last_conv_idx+1:]
        return self

    def init_custom_softmax_layer(self):
        '''
        Method replaces the last dense layer of the VGG
        with a custom softmax layer, with the number of
        units (neurons) equalling the total # of classes
        we find for the given problem.
        :return:
        '''
        self.vgg.finetune(self.train_batches)

        # The above changes the vgg.model, hence reinitialize basic parameters
        self.init_conv_and_fc_models()

        return self

    def init_train_and_val_classes_and_labels(self):
        '''
        Method initialized val and train classes and labels. These properties will be
        used by various other methods.
        :return: self.val_classes, self.val_labels, self.train_classes, self.train_labels,
        '''
        self.val_classes = self.val_batches.classes
        self.val_labels = onehot(self.val_classes)

        self.train_classes = self.train_batches.classes
        self.train_labels = onehot(self.train_classes)

    def tune_softmax_layer_for_epochs(self, num_epochs=4):
        '''
        tune (fit) only the softmax layers. The other layers are made
        non-trainable implicitly.
        :param num_epochs: Specify num_epochs to finetune only the softmax layer
        :return:
        '''

        self.vgg.fit_generator(self.train_batches, self.val_batches, nb_epoch=num_epochs)
        print("Vgg model finetuned.")
        return self;

    def save_model_to_file(self, fileName=None):

        if fileName is None:
            fileName = "weights." + self.runID + ".h5"

        self.vgg.model.save_weights(fileName)
        return self;

    def load_model_from_file(self, fileName=None):

        if fileName is None:
            fileName = "weights." + self.runID + ".h5"

        self.vgg.model.load_weights(fileName)
        return self;

    def build_predictions_on_test_data(self):
        '''
        Method can be used to build predictions, either by default using the underlying
        vgg model, or via any model passed along to it.
        :param model: A Keras model
        :return:
        '''
        test_path = self.data_path + "test"

        b, p= self.vgg.test(test_path, batch_size=2)

        self.prediction = zip([name[8:] for name in b.filenames], p.astype('str'))
        return self;

    def save_predictions_to_file(self, fileName=None):

        if fileName is None:
            fileName = "predictions." + self.runID + ".h5"

        outF = open(fileName, 'w')
        outF.write('image_name,Type_1,Type_2,Type_3\n')

        for elem in self.prediction:
            outF.write(elem[0] + ',' + ','.join(elem[1]) + '\n')
        outF.close()
        return FileLink(fileName)

    def make_linear_layers_trainable(self):
        '''
        Method iterates through the linear layers of the VGG model,
        and sets them as being trainable.
        :return:
        '''
        layers = self.vgg.model.layers
        # Get the index of the first dense layer...
        first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
        print("first dense layer at index: ", first_dense_idx)

        # ...and set this and all subsequent layers to trainable
        for layer in layers[first_dense_idx:]: layer.trainable=True
        print("all dense layers set trainable.")

        return self

    def train_for_epochs(self, num_epochs):
        '''
        Method runs the vgg for specified number of epochs, with the preset learn_rate.
        :param num_epochs:
        :return:
        '''

        self.compile(self.learn_rate)
        self.vgg.fit_generator(self.train_batches, self.val_batches, nb_epoch=num_epochs)
        return self

    def train_rescaled_fc_model_for_epochs(self, num_epochs):
        linearModel = self.get_rescaled_fc_model(new_dropout=self.dropout)
        linearModel.compile(optimizer=Adam(lr=self.learn_rate), loss='categorical_crossentropy', metrics = ['accuracy'])

        linearModel.fit(self.train_precomputed, self.train_labels, batch_size=self.batch_size,
                        nb_epoch= num_epochs, validation_data=(self.val_precomputed, self.val_labels))


        self.init_vgg_with_retrained_fc_layers(linearModel)
        return self

    def compile(self, learn_rate=0.001):
        self.vgg.compile(lr=learn_rate)
        return self


    def init_vgg_with_retrained_fc_layers(self, fc_layers):
        '''
        Method reinitializes the vgg model so that all the layers,
        after the last convolution model, are replaced by the given
        fully connected layers. These fc_layers are typically obtained
        after they were trained separately.
        :param fc_layers: Sequence of keras layers
        :return:
        '''
        self.vgg.model =  self.conv_model
        for layer in fc_layers.layers:
            self.vgg.model.add(layer)

        return self

    def precompute_conv_model_outputs(self):
        '''
        Method precomputes outputs from the convolution models. Initializes the following
        class properties:

        :return: self.val_precomputed, self.train_precomputed
        '''

        print("precomputing conv. model outputs..")

        if(self.conv_model is None):
            self.init_conv_and_fc_models()


        temp_train_batches = self.vgg.get_batches(batch_size=self.batch_size, path=self.train_path, shuffle=False, class_mode=None)
        temp_val_batches = self.vgg.get_batches(batch_size=self.batch_size, path=self.val_path, shuffle=False, class_mode=None)

        self.train_precomputed = self.conv_model.predict_generator(temp_train_batches, self.train_batches.nb_sample)
        self.val_precomputed = self.conv_model.predict_generator(temp_val_batches, self.val_batches.nb_sample)

        self.save_precomputed_conv_models()

        print("done.")
        return self

    def save_precomputed_conv_models(self):
        fName1 = "precomputed_trn_features."+self.runID+".h5"
        fName2 = "precomputed_val_features."+self.runID+".h5"

        save_array(fName1, self.train_precomputed)
        save_array(fName2, self.val_precomputed)
        print("models saved to files: ",fName1, " and ", fName2)

        return self;

    def load_precomputed_conv_models(self):
        '''
        Method loads precomputed conv_model outputs. Initializes:

        :return: self.train_precomputed and self.val_precomputed.
        '''
        print("loading precomputed conv. outputs...")

        fName1 = "precomputed_trn_features."+self.runID+".h5"
        fName2 = "precomputed_val_features."+self.runID+".h5"

        self.train_precomputed = load_array(fName1)
        self.val_precomputed = load_array(fName2)

        # since we're loading precomputed outputs from the conv_model,
        # set this flag to true.
        self.use_precomputed_conv_output = True

        print("done...")
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
        return [o * scale for o in layer.get_weights()]

    def get_rescaled_fc_model(self, new_dropout):
        '''
        prerequisite:
        fc_layers should be a finetuned model from the previous dropout value.
        Hence, ensure that self.vgg has been fully trained/tuned to the dataset.

        :param new_dropout: new dropout probability to keep.
        :return: new fc model whose weights in the fc layers have been rescaled
        '''
        print("getting rescaled fc model...")
        model = self.vgg.get_new_fc_model(self.conv_layers[-1], self.num_softmax_classes, new_dropout)
        
        for l1, l2 in zip(model.layers, self.fc_layers):
            print("found dense layer. Distributing scaled weights..")
            l1.set_weights(self.proc_wgts(l2, self.dropout, new_dropout))

        print ("done...")
        # update dropout
        print("updating dropout from: ",self.dropout," to: ",new_dropout)
        self.dropout = new_dropout

        return model

    def init_and_fit_rescaled_fc_model(self, new_dropout):
        '''
        prerequisites:
        the conv_model should've been used to precompute outputs for both the trn
        and val datasets.

        :param new_dropout: new dropout to apply
        :return: a finely tuned fc model whose weights in the fc layers have been rescaled.
        '''
        print("initializing rescaled fc model...")
        self.rescaled_fc_model = self.get_rescaled_fc_model(new_dropout)

        # such a finely tuned model needs to be updated very slowly...
        opt = Adam(lr=0.000001)
        self.rescaled_fc_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        print("fine tuning the rescaled fc model...")
        self.rescaled_fc_model.fit(self.train_precomputed, self.train_labels, nb_epoch=4,
                       batch_size= self.batch_size, validation_data=(self.val_precomputed, self.val_labels))

        self.rescaled_fc_model.save_weights("rescaled_and_tuned_fc_model."+self.runID+".dropout."+str(new_dropout)+".h5")
        print("done...")

class ExecutorBuilder:

    def __init__(self):
        self.executor = Executor()
        self.train_dense_layers = False

    def and_(self):
        return self

    def with_runID(self, val):
        self.executor.runID = str(val);
        return self

    def with_Vgg16(self):
        self.vgg = Vgg16('vgg16.h5')
        print("Pretrained Vgg16 model loaded.")
        return self

    def with_Vgg16BN(self):
        self.vgg = Vgg16BN()
        print("Pretrained Vgg16BN model loaded.")
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

        # initialize some fundamental class properties
        self.executor.init_validation_and_training_data()
        self.executor.init_train_and_val_classes_and_labels()
        self.executor.init_conv_and_fc_models()
        self.executor.init_custom_softmax_layer()

        if(self.train_dense_layers):
            self.executor.make_linear_layers_trainable()

        #TODO: Figure out why this throws an error!!
        #      self.executor.compile(self.learn_rate)
        return self.executor



if __name__ == "__main__":
    # get parameters from system arguments
    # train_epochs = int(sys.argv[1])
    # learn_rate = float(sys.argv[2])
    # data_path = sys.argv[3]

    # print("train_epochs: ", train_epochs)
    # print("learn_rate", learn_rate)
    # print ("data_path: ", data_path)

    executor = ExecutorBuilder().\
        with_runID("trial").\
        and_().\
        with_Vgg16().\
        and_().\
        train_batch_size(3). \
        and_(). \
        learn_rate(0.001).\
        and_().\
        data_on_path("../data/sample/").\
        build()

    '''------------------------------------------------------------------------------
    NAIVE FIRST ATTEMPT: replace and tune only the softmax layer
    '''
    # executor.tune_softmax_layer_for_epochs(1)


    '''------------------------------------------------------------------------------
    PRECOMPUTE CONV_MODEL OUTPUTS:
    Only precompute outputs from the conv. model and stop.
    '''
    # executor.precompute_conv_model_outputs()

    '''------------------------------------------------------------------------------
    SECOND ATTEMPT:
    1. evaluate and load precomputed conv. model output
    2. train only the linear models for specified # of epochs
    '''
    #executor.load_precomputed_conv_models().and_().train_for_epochs(1).and_().\
    #    build_predictions_on_test_data().and_().save_predictions_to_file("foobar.bn")