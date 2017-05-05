This competition comes with three sets of data: 1) Train.zip 2) Test.zip and 3) Additional.zip

The competition host mentions that the images in "additional.zip" are not of the best quality. 
Upon further examination (sure enough) we find that some of the images are blurry, taken in weird
light conditions, and so on and so forth. Thus, one could try to run the experiments
excluding the "additional" images. I don't know yet for sure what the performance will look like.

Also, the packages Train.zip and Additional.zip come with the images included in one of the three subfolders: Type_1, Type_2, and Type_3.
These are also the categories that we are hoping to classify the images in the "test" image package by.

1. Each of the images in the downloaded packages are some 1000s of pixels wide and high. The first thing we
will do is:

- Copy around 20% of the data to a validation folder (VALID_DATA)
- Use an image segmentation code to zoom in on only the cervical part of the image.
- Crop the image
- Resize it to 224x224
- Save it to a target location

The details for copying images to VALID_DATA path can be found in DataSetup.py. After this, the code in  python (jupyter) 
notebook: "ImageSegmentation_Experiments.ipynb" can be used to figure out how to run the code to segment the images.
Just be sure to provide values for the following parameters:
* TRAIN_DATA, * ADD_DATA, * SAVE_IMG_PATH, * VALID_DATA, (and further down below) * TEST_PATH

The imagess in TRAIN_DATA, ADD_DATA, and VALID_DATA should be arranged in subfolders: Type_1, Type_2, and Type_3.
The target location (SAVE_IMG_PATH) should also have subfolders called Type_1, Type_2 and Type_3. This is the path
from which we will use the images for running our experiments further down the road. 