import numpy as np
from scipy.linalg.blas import sgemm
import time

def f(x):
    if np.isfinite(x):
        return x
    else:
        return 0.0
infToZero = np.vectorize(f)

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

def _computeCorr(F):
    #compute the correlation matrix of filters given
    #an n x k feature response matrix.
    mean = np.mean(F, axis=1)
    std = np.std(F, axis=1)
    n = F.shape[0]
    corr = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            corr[i,j] = np.dot(F[i] - mean[i], F[j] - mean[j]) / (n**2)
    corr /= n
    t = time.time()
    for i in range(n):
        for j in range(n):
            if std[i] == 0 or std[j] == 0:
                corr[i, j] = 0
            else:
                corr[i,j] /= (std[i] * std[j])
    print "normalizing took " + str(time.time() - t)
    return corr

def computeCorr(F):
    #compute the correlation matrix of filters given
    #a mean-centered n x k feature response matrix.
    std = np.std(F, axis=1)
    n = F.shape[0]
    t = time.time()
    corr = sgemm(1.0/n**3, F, F.T)
    #print "computing corr took " + str(time.time() - t)
    t = time.time()
    std = np.matrix(std)
    std = np.dot(std.T, std)
    std = np.asarray(std)
    corr = corr / std
    corr = infToZero(corr)
    """for i in range(n):
        for j in range(n):
            if std[i] == 0 or std[j] == 0:
                corr[i, j] = 0
            else:
                corr[i,j] /= (std[i] * std[j])"""
    #print "normalizing took " + str(time.time() - t)
    return corr

def compareF(F1, F2):
    R1 = computeCorr(F1)
    R2 = computeCorr(F2)
    return np.mean(np.abs(R1 - R2))

def compareImages(l, img1, img2):
    F1 = img1[l]
    F2 = img2[l]

    return compareF(F1, F2)

def compareGroups(l, group1, group2, same):
    n = len(group1)
    dist = np.zeros((n, n))
    avg = 0.0
    count = 0
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            if same and i == j:
                continue
            #print "images ", i, j
            t = time.time()
            d = compareImages(l, group1[i], group2[j])
            #print "comparison took " + str(time.time() - t)
            dist[i,j] = d
            dist[j,i] = d
            avg += d
            count += 1
    avg /= count
    return avg, dist

def compareManyGroups(sn, layer, groupList):
    """
    @param groupList List of lists, where each element is an image location.
    """
    n = len(groupList)
    dist = np.zeros((n, n))
    groupFeatures = []
    for group in groupList:
        F_all = []
        for img in group:
            F = {}
            sn.load_image(img)
            sn.net.forward()
            for l in sn.net.blobs.keys():
                f = sn.getF(l)
                #mean-center to save computation later
                mean = np.mean(f, axis=1)
                f = f - np.asmatrix(mean).T
                f = np.asarray(f)
                F[l] = f
            F_all.append(F)
        groupFeatures.append(F_all)
    inGroup = 0.0
    outGroup = 0.0
    inGroupCt = 0
    outGroupCt = 0
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            #print "groups ", i, j
            avg, d = compareGroups(layer, groupFeatures[i], groupFeatures[j], i == j)
            dist[i,j] = avg
            dist[j,i] = avg
            if i == j:
                inGroup += avg
                inGroupCt += 1
            else:
                outGroup += avg
                outGroupCt += 1

    inGroup /= inGroupCt
    outGroup /= outGroupCt
    np.set_printoptions(precision=4)
    
    print layer
    print "Ingroup average: " + str(inGroup)
    print "Outgroup average: " + str(outGroup)
    print "Ratio: " + str(outGroup / inGroup)
            
def compute():
    import os
    os.environ['GLOG_minloglevel'] = '3'
    import caffe
    caffe.set_device(2)
    caffe.set_mode_gpu()
    import numpy as np
    from simple_net import SimpleNet
    
    sn = SimpleNet()
    imgs = ["dog", "cat", "hedgehog", "rabbit", "squirell", "ball"]
    #imgs = ["dog", "cat"]
    all_imgs = [["categories/" + x + str(i) + ".jpg" for i in range(1, 5)] for x in imgs]

    layers = ["data"]
    #layers = ["conv5_1"]
    for l in layers:
        compareManyGroups(sn, l, all_imgs)

if __name__ == "__main__":
    compute()

