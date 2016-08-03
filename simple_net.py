import os
import sys

import caffe
import numpy as np
from scipy.fftpack import ifftn
from scipy.linalg.blas import sgemm
from scipy.misc import imsave
from scipy.optimize import minimize
from skimage import img_as_ubyte
from skimage.transform import rescale
import time

class SimpleNet:

    def __init__(self):
        caffe.set_mode_cpu()

        model_file = "models/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt"
        pretrained_file = "models/vgg19/VGG_ILSVRC_19_layers.caffemodel"
        mean = np.array([103.939, 116.779, 123.68])

        self._load_model(model_file, pretrained_file, mean)

        def callback(xk):
            if self._callback is not None:
                net_in = xk.reshape(self.net.blobs["data"].data.shape[1:])
                self._callback(self.transformer.deprocess("data", net_in))
        self.callback = callback

    def _load_model(self, model_file, pretrained_file, mean):
        """Load specified model."""
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)

        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        if type(mean) is str:
            transformer.set_mean("data", np.load(mean).mean(1).mean(1))
        else:
            transformer.set_mean("data", mean)
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)

        self.net = net
        self.transformer = transformer

    def _rescale_net(self, img):
        new_dims = (1, img.shape[2]) + img.shape[:2]
        self.net.blobs["data"].reshape(*new_dims)
        self.transformer.inputs["data"] = new_dims

    def load_image(self, imgname):
        img = caffe.io.load_image(imgname)
        self.input_image(img)

    def input_image(self, img):
        self._rescale_net(img)
        net_in = self.transformer.preprocess("data", img)
        self.net.blobs["data"].data[0] = net_in




