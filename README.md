


Introduction
------------
The content in this repository examines the application of several image processing techniques borrowed from deep learning for an image classification problem. The problem is (was) originally taken from Kaggle, and can be found in more detail [here](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening). A copy of the slides summarizing the problem and methodology, that I also presented on a Python meetup, can be found [here](https://www.dropbox.com/s/ioqro3qx41rj2gg/Presentation_Apil_PyData_Meetup.pdf?dl=0).

One of the most important and interesting aspect about this project  is that the images that we need to classify are actually very hard to make any sense out of for me (being a non-expert in the given medical domain). Unlike images of cats, dogs, horses, monkeys, or flowers, which are very easily distinguishable by humans, all the images in the given problem-set are virtually impossible to tell apart to us. And yet, the pre-trained and fine-tuned VGG16 model is able to classify almost 70% of the images in the validation set. This seemingly super-human ability is a huge motivator for me to try understand what exactly is going on, as well as to try and push the limit of conventional deep-learning techniques to get better results.

Environment Setup:
------
The first thing to have is the necessary development environment. The project is based primarily on Python-2, Theano-0.7 and Keras-2.0 libraries. The usage of [anaconda](https://docs.continuum.io/anaconda/) is strongly advised to setup a contained python environment. Once anaconda has been setup, you may use the following commands:

 - To setup a python-2 environment, use **conda create -n my_theano_keras Python=2 **
 - To search for a compatible package of interest, use **conda search -c conda-forge [package-name]**
 - To install the package use **conda install -c [package-name]-[version-number]**


Data Setup:
-----------
Please find the details of setting up data in [src/processing/README.md](https://github.com/apiltamang/kaggle_cervical_cancer_code/tree/master/src/processing)

Prebuilt AWS Image
------------------

Setup weighing you down? It does involve downloading upwards of 50 GBs of data, getting to know the methods and procedures, and upwards of 12 hours to run the provided image segmentation algorithm on all the image-sets. In order to help you get to a speedy start:

1. Open an AWS account, and request for a p2.xlarge instance. The request for a p2.xlarge instance (if you already don't have one) usually takes 2 business days.
2. Click "Launch Instance" button in the EC2 page.
3. Select "Community AMIs", search for **DL RTP Kaggle Cervix Classification**, and hit Select.
4. Go through the necessary configuration steps to get started. This is all still fairly involved, so following a good tutorial to get started on AWS is recommended.

The p2.xlarge instance offers a Tesla k80M GPU, and the above image has all the required libraries configured correctly. Having at least one GPU is indispensable for starting on any deep-learning project. Amazon will charge you $0.90/hr for using/launching the above image. So be sure to shut down your instance when you no longer need it.

Experiments
-------------
At the moment of this writing, it is impossible for me to list out exactly what has been done, or what will be accomplished. This is very much an experimental (and research-ish) work in progress. So far so good, I have only managed to run some classification experiments using the pretrained VGG16 model. The file (src/Experiments With VGG16.ipynb) lists out the basic training procedures.


A classification experiment, in the intended state of things, would begin with the following line:
```
var executor = ExecutorBuilder().\
    with_runID("second").\
    and_().\
    with_Vgg16().\
    and_().\
    train_batch_size(128). \
    and_(). \
    learn_rate(0.001).\
    and_().\
    data_on_path("../data/full/").\
    build()
```
The following line:
```
executor.train_for_epochs(5)
``` 
would now train this model for the specified number of epochs. Likewise, the line
 ```
executor.build_predictions_on_test_data().and_().save_predictions_to_file("test-predictions")
``` 
allows you to run the model against the test data and generate predictions. The full gamut of everything you can do with the **Executor** class can be found by inspecting the class itself, or looking at sample code in src/Experiments With VGG16.ipynb. More details on the methodology employed can be found by looking at the first lecture in [fast.ai](http://course.fast.ai/lessons/lesson1.html), or this [lecture post](http://cs231n.github.io/transfer-learning/). This methodology is known both as *transfer learning* or *fine tuning* and serves as a good base-line for any more advanced classification method.

Work In Progress
----------------
1. Implementing the fine-tuning methodology for the ResNet50 architecture. The job has become considerably hard: given that ResNet50 had to build from a functional Keras model (versus a Sequential Keras model used for VGG16). 
2. Migrate the keras code for VGG16 from Keras 1.x to Keras 2.0. This will surely take a behemoth of time in and by itself.
3. Make predictions using an ensemble of methods. 

More ambitious plans

 1. Use a one-shot learning approach for image classification. The
    images provided for this problem is limited, and many of them are
    blurry and just bad samples.
 2. Other techniques such as maybe Generative Adversarial Networks (GANs)


 
