import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
#import cv2
import logging
import os.path

from pathlib import Path
#from filelock import FileLock, Timeout
import tensorflow as tf

logger = logging.getLogger(__name__)

def load_sample(image, groundtruth):
    _shape = (1, 480, 640)

    batch = {}
    img = imread(image)

    if len(img.shape) == 2:
        img = img[np.newaxis, :]
    if img.shape != _shape:
        logger.warning("shape_mismatch, resizing image shape to: {}".format(_shape))
        img = resize(img, _shape)
    img = img.astype(np.float32)
    img -= 128
    img /= 128
    batch['data'] = img

    batch['label'] = np.asarray([int(image.split('/')[-2].split('_', maxsplit=1)[0])-1])

    gt = imread(groundtruth)
    if len(gt.shape) == 2:
        gt = gt[np.newaxis,:]
    if gt.shape != _shape:
        gt = resize(gt, _shape, order=0)
    batch['seg'] = gt

    assert batch['data'].shape == batch['seg'].shape
    assert len(batch['label'].shape) == 1
    assert batch['data'].shape == _shape
    assert batch['seg'].shape == _shape

    _mask = np.invert(batch['seg'].astype(np.bool_))

    batch['data'][_mask] = 0

    return batch


def load_sample_cgan(mask, image2, image1):

    assert os.path.isfile(image1)
    assert os.path.isfile(image2)
    assert os.path.isfile(mask)

    batch = {}

    img1 = imread(image1).astype(np.float32)
    img1 = np.moveaxis(img1, -1, 0)

    img1 = img1 / 127.5 - 1
    img2 = imread(image2).astype(np.float32)
    img2 = rgb2gray(img2)
    img2 = img2.reshape(1, *img2.shape)
    img2 = img2 / 127.5 - 1

    img3 = imread(mask).astype(np.float32)
    img3 = rgb2gray(img3)
    img3 = img3.reshape(1, *img3.shape)
    img3 = img3 / 127.5 - 1

    img = np.concatenate((img1, img2), axis=0)
    batch['data'] = img

    img3[(img3 > -1.) & (img3 < 1.)] = 0    ## To replace with 'Dont care' labels for LW data

    batch['seg'] = img3


    return batch


def load_sample_cgan_test(mask, image2, image1):

    assert os.path.isfile(mask)

    batch = {}

    img3 = imread(mask).astype(np.float32)
    img3 = rgb2gray(img3)

    batch['data'] = np.zeros((4, *img3.shape), dtype=np.float32)

    img3 = img3.reshape(1, *img3.shape)
    img3 = img3 / 127.5 - 1

    batch['seg'] = img3

    return batch