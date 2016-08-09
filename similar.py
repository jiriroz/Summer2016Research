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
from skimage.transform import resize
import time
import json
from simple_net import SimpleNet

VGG_LAYERS = [
"conv1_1",
"conv1_2",
"conv2_1",
"conv2_2",
"conv3_1",
"conv3_2",
"conv3_3",
"conv3_4",
"conv4_1",
"conv4_2",
"conv4_3",
"conv4_4",
"conv5_1",
"conv5_2",
"conv5_3",
"conv5_4"
]

layers = [
"conv3_4",
"conv4_1",
"conv4_2",
"conv4_3",
"conv4_4",
"conv5_1",
"conv5_2",
"conv5_3",
"conv5_4"
]

CUTOFF = 0.5

def perceptualComparison(infile):
    with open(infile, "r") as f:
        imNames = f.read().split("\n")
    images = []
    for im in imNames:
        try:
            img = caffe.io.load_image(im)
            images.append(img)
        except Exception as e:
            print "Could not read image " + im
            print e

    net = SimpleNet()
    G_layers = {layer:[] for layer in layers}
    for img in images:
        net.input_image(img)
        net.net.forward()
        for layer in layers:
            F = net.net.blobs[layer].data[0].copy()
            F.shape = (F.shape[0], -1)
            G_layers[layer].append(sgemm(1, F, F.T))
    
    G_avg = {layer: np.sum(G_layers[layer], axis=0)/len(G_layers[layer])
             for layer in G_layers}
    loss = {}
    for layer in G_layers:
        loss[layer] = np.sum([(G_img - G_avg[layer])**2
                      for G_img in G_layers[layer]], axis=0)

    indices = {}
    for layer in loss:
        n = loss[layer].shape[0]
        indices[layer] = np.dstack(np.unravel_index(np.argsort(loss[layer].ravel()), (n, n)))
        indices[layer] = indices[layer][:int(n*n*CUTOFF)]

    masks = {}
    for layer in indices:
        n = loss[layer].shape[0]
        masks[layer] = np.zeros((n, n))
        for ind in indices[layer]:
            masks[ind[0], ind[1]] = 1
    
        


def main():
    if sys.argv[1] == "-1":
        caffe.set_mode_cpu()
    else:
        gpu = int(sys.argv[1])
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    infile = sys.argv[2]
    perceptualComparison(infile)


if __name__ == "__main__":
    main()

