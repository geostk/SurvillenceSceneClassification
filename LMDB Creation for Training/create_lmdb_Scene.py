'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Muhammad Shahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = 'E:/Codding/MIT_Places_Modified/Scene16Classes2k/train_lmdb'
validation_lmdb = 'E:/Codding/MIT_Places_Modified/Scene16Classes2k/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
#os.system('rm -rf  ' + validation_lmdb)


train_data = [img for img in glob.glob("../DataScene/*jpg")]
#test_data = [img for img in glob.glob("../input/test1/*jpg")]

#Shuffle train_data
random.shuffle(train_data)

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(7e9))
with in_db.begin(write=True) as in_txn:
    print 'Sucess 1'
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  6 == 0:
            continue
            print(in_idx)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'Building' in img_path:
            label = 0
        elif 'cafeteria' in img_path:
              label = 1
        elif 'classroom' in img_path:
              label = 2
        elif 'conference_room' in img_path:
              label = 3
        elif 'downtown' in img_path:
              label = 4 
        elif 'driveway' in img_path:
              label = 5	
        elif 'Highway_Road' in img_path:
              label = 6
        elif 'hospital_room' in img_path:
              label = 7
        elif 'office' in img_path:
              label = 8
        elif 'Park' in img_path:
              label = 9	
        elif 'Parking' in img_path:
              label = 10
        elif 'parking_underground' in img_path:
              label = 11
        elif 'platform' in img_path:
              label = 12
        elif 'ShoppingMall' in img_path:
              label = 13
        elif 'Street' in img_path:
              label = 14
        elif 'supermarket' in img_path:
              label = 15
        print(in_idx)
        print(label)
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx + label * 2000), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()  
print '\nCreating validation_lmdb'

# in_db = lmdb.open(validation_lmdb, map_size=int(1e9))
# with in_db.begin(write=True) as in_txn:
    # print 'Sucess 1'
    # for in_idx, img_path in enumerate(train_data):
        # if in_idx %  6 != 0:
            # continue
            
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'Building' in img_path:
            # label = 0
        # elif 'cafeteria' in img_path:
              # label = 1
        # elif 'classroom' in img_path:
              # label = 2
        # elif 'conference_room' in img_path:
              # label = 3
        # elif 'downtown' in img_path:
              # label = 4 
        # elif 'driveway' in img_path:
              # label = 5	
        # elif 'Highway_Road' in img_path:
              # label = 6
        # elif 'hospital_room' in img_path:
              # label = 7
        # elif 'office' in img_path:
              # label = 8
        # elif 'Park' in img_path:
              # label = 9	
        # elif 'Parking' in img_path:
              # label = 10
        # elif 'parking_underground' in img_path:
              # label = 11
        # elif 'platform' in img_path:
              # label = 12
        # elif 'ShoppingMall' in img_path:
              # label = 13
        # elif 'Street' in img_path:
              # label = 14
        # elif 'supermarket' in img_path:
              # label = 15
        # print(in_idx)
        # print(label)
        # datum = make_datum(img, label)
        # in_txn.put('{:0>5d}'.format(in_idx + label * 2000), datum.SerializeToString())
        # print '{:0>5d}'.format(in_idx) + ':' + img_path
# in_db.close()  
