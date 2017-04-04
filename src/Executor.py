from keras.layers import Dense
from keras.optimizers import Adam

from Vgg16 import Vgg16
from IPython.display import FileLink

class Executor:

    def __init(self):
        self.vgg = None
        self.batch_size = None
        self.train_batches = None
        self.val_batches = None
        self.data_path = None
        self.learn_rate = None

    def and_(self):
        return self;

    def set_Vgg(self, vgg):
        self.vgg = vgg

    def init_validation_and_training_data(self):
        train_path = self.data_path+"train"
        val_path = self.data_path+"valid"

        self.train_batches = self.vgg.get_batches(batch_size=self.batch_size, path=train_path)
        self.val_batches = self.vgg.get_batches(batch_size=self.batch_size, path=val_path)

        print("initialized training data from: "+train_path)
        print("initialized validation data from: "+val_path)

    def finetune_only_softmax_layer_for_epochs(self, num_epochs):
        self.vgg.finetune(self.train_batches)

        self.vgg.fit_generator(self.train_batches, self.val_batches, nb_epoch=num_epochs)
        print("Vgg model finetuned.")
        return self;

    def save_model_to_file(self, fileName):
        self.vgg.model.save_weights(fileName)
        return self;

    def load_model_from_file(self, fileName):
        self.vgg.model.load_weights(fileName)
        return self;

    def build_predictions_on_test_data(self):
        test_path = self.data_path + "test"
        b, p= self.vgg.test(test_path, batch_size=2)

        self.prediction = zip([name[8:] for name in b.filenames], p.astype('str'))
        return self;

    def save_predictions_to_file(self, fileName):
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

        self.executor.compile(self.learn_rate)
        return self.executor


if __name__ == "__main__":
    executor = ExecutorBuilder().\
        with_Vgg16().\
        and_().\
        train_batch_size(2). \
        and_(). \
        learn_rate(0.001).\
        and_().\
        data_on_path("../data/sample/").\
        and_().\
        trainable_linear_layers().\
        build()

    executor.finetune_only_softmax_layer_for_epochs(1).and_().save_model_to_file("weights.trial.h5").and_().\
        build_predictions_on_test_data().and_().save_predictions_to_file("predictions.trial.h5")
