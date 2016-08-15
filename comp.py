import numpy as np
from scipy.linalg.blas import sgemm
import time

IN = np.array([])
OUT = np.array([])

def computeReprs(sn, all_imgs):
    """Given a list of groups where each contains a list of image names,
    return feature representations and feature covariances of the corresponding
    images."""
    groupFeatures, groupCovs = [], []
    for group in all_imgs:
        F_all = []
        cov_all = []
        for img in group:
            F = {}
            cov = {}
            sn.load_image(img)
            sn.net.forward()
            for l in sn.net.blobs.keys():
                f = sn.getF(l)
                mean = np.mean(f, axis=1)
                f_center = f - np.asmatrix(mean).T
                f_center = np.asarray(f_center)
                F[l] = f_center
                n = f.shape[0]
                cov[l] = sgemm(1.0/n, f_center, f_center.T)
            F_all.append(F)
            cov_all.append(cov)
        groupFeatures.append(F_all)
        groupCovs.append(cov_all)
    return groupFeatures, groupCovs

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

def compareNorm(sn, img1, img2, l):
    G1 = np.log(getG(sn, img1, l))
    G2 = np.log(getG(sn, img2, l))
    G1 = G1 / np.sum(G1)
    G2 = G2 / np.sum(G2)
    n = G1.shape[0]
    return np.sum(np.abs(G1 - G2)) / n**2

def computeCov(F):
    #Compute the covariance matrix of filters given
    #a mean-centered n x k feature response matrix.
    n = F.shape[0]
    cov = sgemm(1.0/n, F, F.T)
    return cov

def computeCorr(F):
    #compute the correlation matrix of filters given
    #a mean-centered n x k feature response matrix.
    std = np.std(F, axis=1)
    n = F.shape[0]
    corr = sgemm(1.0/n**3, F, F.T)
    std = np.matrix(std)
    std = np.dot(std.T, std)
    std = np.asarray(std)
    corr = corr / std
    corr = infToZero(corr)
    return corr

def compareF(F1, F2, compMethod):
    cov1 = compMethod(F1)
    cov2 = compMethod(F2)
    return compareCov(cov1, cov2)

def compareCov(cov1, cov2):
    return np.mean(np.abs(cov1 - cov2))

def compareImagesGivenFeatures(l, img1, img2):
    F1 = img1[l]
    F2 = img2[l]
    return compareF(F1, F2, computeCov)

def compareImagesGivenCovariances(l, img1, img2):
    cov1 = img1[l]
    cov2 = img2[l]
    return compareCov(cov1, cov2)

def compareGroups(l, group1, group2, same):
    global IN
    global OUT
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
            t = time.time()
            #d = compareImagesGivenFeatures(l, group1[i], group2[j])
            d = compareImagesGivenCovariances(l, group1[i], group2[j])
            dist[i,j] = d
            dist[j,i] = d
            avg += d
            count += 1
            if same:
                IN = np.append(IN, d)
            else:
                OUT = np.append(OUT, d)
    avg /= count
    return avg, dist

def compareManyGroups(sn, layer, groupFeatures):
    """
    @param groupFeatures list of lists where each element is a dictionary of
    feature representations for each layer for a particular image
    """
    n = len(groupFeatures)
    dist = np.zeros((n, n))
    inGroup = 0.0
    outGroup = 0.0
    inGroupCt = 0
    outGroupCt = 0
    t = time.time()
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
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
    print "Ratio: " + str(outGroup / inGroup)
    stdIn = np.std(IN)
    stdOut = np.std(OUT)
    distIn = (np.mean(OUT) - np.mean(IN)) / stdIn
    distOut = (np.mean(OUT) - np.mean(IN)) / stdOut
    print "In mean:", np.mean(IN)
    print "Out mean", np.mean(OUT)
    print "In std: ", stdIn
    print "Normalized distance in: ", distIn

    T = time.time() - t
    n = len(IN) + len(OUT)
    print "Complete comparison took " + str(T)
    print "Performed " + str(n) + "comparisons."
    print "Average comparison took " + str(T/n)

    print ""
            
def compute():
    import os
    os.environ['GLOG_minloglevel'] = '3'
    import caffe
    caffe.set_device(3)
    caffe.set_mode_gpu()
    import numpy as np
    from simple_net import SimpleNet
    
    sn = SimpleNet()
    imgs = ["dog", "cat", "hedgehog", "rabbit", "squirell", "ball"]
    all_imgs = [["categories/" + x + str(i) + ".jpg" for i in range(1, 5)] for x in imgs]

    #layers = ["conv4_1", "conv4_2", "conv4_3", "conv4_4" ,"conv5_1", "conv5_2", "conv5_3", "conv5_4"]
    layers = ["conv5_4"]
    groupFeatures, groupCovs = computeReprs(sn, all_imgs)
    for l in layers:
        IN = np.array([])
        OUT = np.array([])
        compareManyGroups(sn, l, groupCovs)

if __name__ == "__main__":
    compute()

