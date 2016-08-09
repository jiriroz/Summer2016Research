from simple_net import SimpleNet
import sys
from scipy.linalg.blas import sgemm
import matplotlib.pyplot as plt
import numpy as np

def visualize():
    net = SimpleNet()
    img = sys.argv[1]
    
    net.load_image(img)
    
    net.net.forward()

    k = 1
    i = 0
    
    for layer in sys.argv[2:]:
        i += 1
        F = net.net.blobs[layer].data[0].copy()
        F.shape = (F.shape[0], -1)
        G = sgemm(1, F, F.T) / 1e9
        std = np.std(G)
        G /= std

        plt.subplot(k, k, i)
        n, bins, patches = plt.hist(G.flatten(), 100, normed=1, facecolor='green', alpha=0.75)
        
        plt.xlabel('')
        plt.ylabel('')
        plt.title(layer)
    
    plt.show()

def getG(net, layer, img):
    net.load_image(img)
    net.net.forward()
    F = net.net.blobs[layer].data[0].copy()
    F.shape = (F.shape[0], -1) 
    G = sgemm(1, F, F.T) / 1e9
    std = np.std(G)
    G /= std
    return G

def multiply():
    net = SimpleNet()
    img1 = sys.argv[1]
    img2 = sys.argv[2]
    layer = sys.argv[3]

    G1 = getG(net, layer, img1)
    G2 = getG(net, layer, img2)

    G = np.multiply(G1, G2)

    #plt.subplot(k, k, i)
    n, bins, patches = plt.hist(G.flatten(), 100, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('')
    plt.ylabel('')
    plt.title(layer)

    plt.show()

if __name__ == "__main__":
    multiply()
    #visualize()
