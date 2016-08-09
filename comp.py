import numpy as np
from scipy.linalg.blas import sgemm

def getGSimple(sn, img, l):
    sn.load_image(img)
    sn.net.forward()
    F = sn.net.blobs[l].data[0].copy()
    F.shape = (F.shape[0], -1)
    return sgemm(1, F, F.T)

def getG(sn, img, l):
    sn.load_image(img)
    sn.net.forward()
    F = sn.net.blobs[l].data[0].copy()
    F.shape = (F.shape[0], -1)
    sc = 1.0/F.shape[1]
    norm = np.apply_along_axis(np.linalg.norm, 1, F)
    Fnorm = F / np.matrix(norm).T
    return sgemm(sc, Fnorm, Fnorm.T)

def compare(sn, img1, img2, l):
    G1 = getG(sn, img1, l)
    G2 = getG(sn, img2, l)
    n = G1.shape[0]
    return np.sum(np.square(G1 - G2)) / n**2 

def computeNorm(sn, img1, img2, l):
    G1 = np.log(getG(sn, img1, l))
    G2 = np.log(getG(sn, img2, l))
    G1 = G1 / np.sum(G1)
    G2 = G2 / np.sum(G2)
    n = G1.shape[0]
    return np.sum(np.abs(G1 - G2)) / n**2

def computeCorr(F):
    #compute the correlation matrix of filters given
    #an n x k feature response matrix.
    mean = np.mean(F, axis=1)
    std = np.std(F, axis=1)
    n = F.shape[0]
    corr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            corr[i,j] = np.dot(F[i] - mean[i], F[j] - mean[j]) / (n**2)
    #return corr
    corr /= n
    for i in range(n):
        for j in range(n):
            if std[i] == 0 or std[j] == 0:
                corr[i, j] = 0
            else:
                corr[i,j] /= (std[i] * std[j])
    return corr










