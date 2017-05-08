import cv2
from src.processing.DataSetup import DataSetup
from src.processing.ImageIterator import ImageIterator


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def mark_image(image):
    threshold_value = 35

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    blurry = False

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < threshold_value:
        blurry = True

    return image,blurry


if __name__ == "__main__":

    # create folders
    a = DataSetup()
    a.create_folders("../../data/marked/")

    # Process images in the train and validation set.
    TRAIN_PATH =  "/Users/apil.tamang/kaggle_cervical_cancer_code/data/segmented/train"
    VALID_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/segmented/valid"
    TEST_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/segmented/test"
    SAVE_IMG_PATH = "/Users/apil.tamang/kaggle_cervical_cancer_code/data/marked"
    b = ImageIterator(TRAIN_PATH, VALID_PATH, TEST_PATH, SAVE_IMG_PATH)

    b.process_train_images(mark_image)



