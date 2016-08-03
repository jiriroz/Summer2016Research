import os
import sys

import caffe
import numpy as np
from scipy.linalg.blas import sgemm
from scipy.misc import imsave
from scipy.optimize import minimize
from skimage import img_as_ubyte
from skimage.transform import rescale
import time

def computeStyle(simplenet, imgname):
    """
    Compute style representations of an image for every layer.
    @param simplenet SimpleNet instance
    @param imgname
    @return dict where keys are layers names and values gram matrices
    """
    simplenet.load_image(imgname)
    net = simplenet.net
    net.forward()
    reprs = dict()
    for layer in net.blobs.keys()[1:]: #skip data
        F = net.blobs[layer].data[0].copy()
        F.shape = (F.shape[0], -1)
        reprs[layer] = sgemm(1, F, F.T)
    return reprs



