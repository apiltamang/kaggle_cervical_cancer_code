import os
import random

import sys
from scipy.misc import imread,imresize,imsave    # I have scipy 0.19.0

class DataSetup:

    def __init__(self):
        self.classes = ['Type_1', 'Type_2', 'Type_3']

    def create_folders(self, dat_root):
        dirs = ['train','valid']

        # create train and valid folders
        for dir in dirs:
            for typ in self.classes:
                need_dir = os.path.join(dat_root, dir, typ)

                if not os.path.exists(need_dir):
                    os.makedirs(need_dir)

        # create test folder
        test_dir = os.path.join(dat_root, "test", "unknown")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

    def create_valid_imgs_list(self, imgs_src, ext_name):
        random.seed(20170316)

        for typ in self.classes:

            # call ext_name with values ".additional.txt" AND ".txt"

            with open(typ + ext_name, "w") as validF:
                for f in os.listdir(os.path.join(imgs_src, typ)):
                    if f.endswith("jpg") and random.random() >= 0.8:
                        validF.write(f + "\n")

    ### Downsize image to 224 * 224 * 3 ###
    def resize(self, img_path):
        img = imread(img_path)
        img = imresize(img, (224,224,3))
        return img

    ### Decide whether the image gets saved to train or valid ###
    def save_train_and_val_imgs_to_path(self, val_imgs_list_file_path, img_path, img, typ, imgs_dest_root):
        img_name = os.path.basename(img_path)

        if(typ not in self.classes):
            sys.exit("Invalid typ argument")

        valid = open(val_imgs_list_file_path).read().split()

        if(img_name in valid):
            new_path = os.path.join(imgs_dest_root + "valid", typ, img_name)
        else:
            new_path = os.path.join(imgs_dest_root + "train", typ, img_name)
        imsave(new_path, img)

    # init train and validation data sets
    def init_train_and_valid_set(self, imgs_src, imgs_dest, val_imgs_list_file_ext):

        for typ in self.classes:
            val_imgs_list_file = typ + val_imgs_list_file_ext

            # call method with:  img_src_path = ../../download/train/, ../../download/additional/
            print("=====================================")
            imgs_src_typ = os.path.join(imgs_src, typ)
            print("processing images in path: ",imgs_src_typ)

            tot_files = len(os.listdir(imgs_src_typ))
            counter = 0;

            for f in os.listdir(imgs_src_typ):

                counter=counter+1
                frac = int(float(counter)/tot_files * 100.)," %"
                sys.stdout.write("progress: %s  \r" % (str(frac)))
                sys.stdout.flush()

                if f.endswith("jpg"):

                    try:
                        impath = os.path.join(imgs_src_typ, f)

                        #print "processing: ",impath
                        img = self.resize(impath)
                        self.save_train_and_val_imgs_to_path(val_imgs_list_file, impath, img, typ, imgs_dest)
                    except:
                        print "failed processing: ",impath
                        pass
            print("=========== done ====================")

    ### test
    def process_test(self, imgs_src, imgs_dest):

        imgs_dest = os.path.join(imgs_dest, "test/unknown/")
        print("=====================================")
        print "processing images in path: ",imgs_src
        tot_files = len(os.listdir(imgs_src))
        counter = 0;

        for f in os.listdir(os.path.join(imgs_src)): # imgs_src = "../../download/test/"
            if f.endswith("jpg"):
                impath = os.path.join(imgs_src, f)

                counter=counter+1
                frac = int(float(counter)/tot_files * 100.)," %"
                sys.stdout.write("progress: %s  \r" % (str(frac)))
                sys.stdout.flush()

                img = self.resize(impath)
                # imgs_dest = "../../test/unknown/
                imsave(os.path.join(imgs_dest, os.path.basename(impath)), img)
        print("=========== done ====================")

if __name__ == "__main__":
    a = DataSetup()
    a.create_folders("../../data/full/")

    # first process the testing data-set
    # a.process_test(imgs_src="../../download/test/", imgs_dest="../../data/full/")

    # # process and split data in /download/train directory. Resizes, splits and saves to folders: imgs_dest/train, imgs_dest/valid
    # a.create_valid_imgs_list(imgs_src="../../download/train/", ext_name=".txt")
    # a.init_train_and_valid_set(imgs_src="../../download/train/", imgs_dest="../../data/full/", val_imgs_list_file_ext=".txt")

    # process and split data in /download/additional directory. Resizes, splits and saves to folders: imgs_dest/train, imgs_dest/valid
    a.create_valid_imgs_list(imgs_src="../../download/additional/", ext_name=".additional.txt")
    a.init_train_and_valid_set(imgs_src="../../download/additional/", imgs_dest="../../data/full/", val_imgs_list_file_ext=".additional.txt")


