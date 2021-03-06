from src.processing.DataSetup import DataSetup
from src.processing.ImageIterator import ImageIterator
import numpy as np
import cv2

def normalize_pixels(x):
    save = True
    #vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))
    #x = x - vgg_mean
    mean_red = 123.68
    mean_green = 116.779
    mean_blue = 103.939

    vgg_mean = np.array([mean_red, mean_green, mean_blue], dtype=np.float32).reshape(1,1,3)

    x = x - vgg_mean
    return x,save

def hsv_transform(x):
    save = True
    image = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

    return image, save

def Ycr_transform(x):
    save = True
    image = cv2.cvtColor(x, cv2.COLOR_BGR2YCR_CB)

    return image, save

def clahe_transform(x):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(x)

    return image, True

if __name__ == "__main__":

    # create folders
    a = DataSetup()
    a.create_folders("../../data/color_transform/")

    # Process images in the train and validation set.
    TRAIN_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/segmented/train"
    VALID_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/segmented/valid"
    TEST_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/segmented/test"
    SAVE_IMG_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/color_transform"
    b = ImageIterator(TRAIN_PATH, VALID_PATH, TEST_PATH, SAVE_IMG_PATH)

    # see the image as being pixels normalized
    #b.process_train_images(normalize_pixels)

    # transform the image into different spectrum
    # b.process_train_images(hsv_transform)

    b.process_valid_images(hsv_transform)
    # b.process_train_images(clahe_transform)

