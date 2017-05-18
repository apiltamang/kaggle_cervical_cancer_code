

Data Setup
----------

You aren't going to do any image classifications until you have the images, so let's get them downloaded and stored somewhere. Keras expects images classes be grouped by the directories. As an example, for this project, you'd want a folder that looks as the follows before any training can begin:

![enter image description here](https://lh3.googleusercontent.com/IdDkXAk7i9vVrBGGDVoZmyxff04U7S5dI0-amvCxez9nPI-rxd4AmtgxU-WGqoeMpUHWrNWs=s600 "Screenshot 2017-05-15 17.01.37.png")

Downloading and Prepping Images
-------------------------------
This competition comes with three sets of data. Go ahead and download them and from here, and unpack them into a folder.

1. Train.zip 
2. Test.zip
3. Additional.zip

**Background:**
The competition host mentions that the images in "additional.zip" are not of the best quality. Upon further examination (sure enough) we find that some of the images are blurry, taken in weird light conditions, and so on and so forth. Thus, one could try to run the experiments excluding the "additional" images. I don't know yet for sure what the performance will look like.

The images in the downloaded packages are some 1000s of pixels wide and high. The ways in which we will proceed is:

 1. Copy around 20% of the data to a validation folder (VALID_DATA),
 2. Use an image segmentation code to zoom in on only the cervical part of the image,
 3. Crop the image so that the cervical parts are zoomed in roughly centered, and
 4. Resize it to 224x224 and save to a target location.

The details of achieving the above should be looked in the file: *DataSetup.py*. There is more code in jupyter notebook: "ImageSegmentation_Experiments.ipynb" which can be used to figure out how to run the code to segment the images. Just be sure to provide the right values image locations.

Additional Image Processing
---------------------------
There are additional scripts and notebooks (.ipynb) included in [src/processing](https://github.com/apiltamang/kaggle_cervical_cancer_code/tree/master/src/processing) that can be used to pre-process the images in many different ways. In particular, you can use the **ImageIterator.py** class to pass in a function that applies any custom preprocessing to the images, and save it in a suitable target folder. For instances, code can be found that

 - Takes the original images, applies a segmentation algorithm to zoom in onto the cervical section, crops the image, and save to the right *Type* (Type_1, Type_2, Type_3 )folder in the right data *category* (i.e. train, valid, or test) folder.
 - Computes an un-blurriedness factor, and only saves images who un-blurriedness exceeds a specified threshold (with the aim to remove images that are more blurried than the specified threshold).
 - Computers the HSV color-spatial transformation.

The above list is only some examples of some simple image processing I sought to try. Feel free, by all means, to compute your own preprocessing. All this could still have been achieved using Kera's [ImageDataGenerator](https://keras.io/preprocessing/image/), if I remember correctly, but I chose this for a specific reason that I will allude to later.
