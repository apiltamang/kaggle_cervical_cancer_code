import os, sys
from glob import glob
import numpy as np
from src.processing.ImageSegmentation import get_image_data
import cv2

class ImageIterator:

    def __init__(self, train_path, valid_path, test_path, save_img_path):
        self.TRAIN_DATA= train_path
        self.VALID_DATA= valid_path
        self.TEST_DATA = test_path
        self.SAVE_IMG_PATH = save_img_path

    def getType_1_2_3_ids(self, train_path):
        print("looking for images in: ", train_path)
        type_1_files = glob(os.path.join(train_path, "Type_1", "*.jpg"))
        type_1_ids = np.array([s[len(os.path.join(train_path, "Type_1")) + 1:-4] for s in type_1_files])
        type_2_files = glob(os.path.join(train_path, "Type_2", "*.jpg"))
        type_2_ids = np.array([s[len(os.path.join(train_path, "Type_2")) + 1:-4] for s in type_2_files])
        type_3_files = glob(os.path.join(train_path, "Type_3", "*.jpg"))
        type_3_ids = np.array([s[len(os.path.join(train_path, "Type_3")) + 1:-4] for s in type_3_files])

        # you may do the following to process
        # a fraction of the data for early experimentation.

        #type_1_ids = type_1_ids[:10]
        #type_2_ids = type_2_ids[:10]
        #type_3_ids = type_3_ids[:10]

        return type_1_ids, type_2_ids, type_3_ids


    def process_and_save_image(self, processing_func, src_image_path, save_img_path, type_ids_list, image_type):
        for k, type_ids in enumerate(type_ids_list):
            m = len(type_ids)
            train_ids = sorted(type_ids)
            counter = 0
            good_images = 0

            for i in range(m):
                image_id = train_ids[counter]
                try:

                    counter += 1

                    # Get the image
                    if image_type == "TRAIN" or image_type == "VALID":
                        img = get_image_data(image_id, 'Type_%i' % (k + 1), src_image_path)
                    else:
                        img = get_image_data(image_id, 'unknown', src_image_path)

                    if img is None:
                        continue

                    # process the image using supplied func
                    img, save = processing_func(img)

                    # determine whether to save processed image or not
                    if save:
                        good_images += 1
                        if (image_type == "TRAIN" or image_type == "VALID"):
                            self.save_image(image_id, 'Type_%i' % (k + 1), save_img_path, img)
                        else:
                            self.save_image(image_id, 'unknown', save_img_path, img)

                    # display percentage statistics
                    frac = int(float(counter) / m * 100.), " %"
                    sys.stdout.write("progress: %s  \r" % (str(frac)))
                    sys.stdout.flush()

                except Exception as err:
                    print("error processing image: ", image_id, '. Type_%i' % (k + 1), " on path: ", src_image_path)
                    print("============================")
                    print(err)
                    print("============================")

            print("moved: ", good_images, " out of: ", counter, " for Type_%i" % (k + 1))

    def save_image(self, image_id, image_type, DATA_ROOT, img):
        data_path = os.path.join(DATA_ROOT, image_type)
        ext = 'jpg'
        img_path = os.path.join(data_path, "{}.{}".format(image_id, ext))
        cv2.imwrite(img_path, img)


    def process_train_images(self, processing_func):
        # process image in the train directory in download/
        print("data in: ", self.TRAIN_DATA)

        all_type_ids = self.getType_1_2_3_ids(self.TRAIN_DATA)
        save_img_path = os.path.join(self.SAVE_IMG_PATH,"train")

        self.process_and_save_image(processing_func, self.TRAIN_DATA, save_img_path, all_type_ids, image_type="TRAIN")


    def process_valid_images(self, processing_func):

        # process image in the train directory in download/
        print("data in: ", self.VALID_DATA)

        all_type_ids = self.getType_1_2_3_ids(self.VALID_DATA)
        save_img_path = os.path.join(self.SAVE_IMG_PATH,"valid")

        self.process_and_save_image(processing_func, self.VALID_DATA, save_img_path, all_type_ids, image_type="VALID")

    def process_test_images(self, processing_func):

        print("data in: ", self.TEST_DATA)

        test_files = glob(os.path.join(self.TEST_DATA, "unknown", "*.jpg"))
        test_ids = np.array([s[len(os.path.join(self.TEST_DATA, "unknown")) + 1:-4] for s in test_files])

        save_img_path = os.path.join(self.SAVE_IMG_PATH, "test")
        self.process_and_save_image(processing_func, self.TEST_DATA, save_img_path, [test_ids], image_type="TEST")
