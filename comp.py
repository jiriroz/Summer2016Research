import os
import sys
import random
import numpy as np
from scipy.linalg.blas import sgemm
import time

#os.environ['GLOG_minloglevel'] = '3'
import caffe
import numpy as np
from simple_net import SimpleNet


"""
Running time: n * (feedforward time) + n*n/2 * (comparison time)
comparison time = 0.0002 s

"""

def getImages(path, n):
    """Randomly choose n images from the given directory
    and return the list of their locations.
    """
    images = os.listdir(path)
    return random.sample(images, n)

def chooseImages(path, n):
    """Path contains directories, each containing one category."""
    categories = os.listdir(path)
    for cat in categories:
        catpath = path + "/" + cat
        _catpath = path + "/_" + cat
        os.makedirs(_catpath)
        imgs = getImages(catpath, n)
        for img in imgs:
            impath = catpath + "/" + img
            newimpath = _catpath + "/" + img
            os.system("cp {} {}".format(impath, newimpath))
            

def computeReprs(sn, all_imgs):
    """Given a list of groups where each contains a list of image names,
    return feature representations and feature covariances of the corresponding
    images."""
    i = 1
    groupFeatures, groupCovs = [], []
    for group in all_imgs:
        print "Group " + str(i)
        i += 1
        F_all = []
        cov_all = []
        for img in group:
            F = {}
            cov = {}
            sn.load_image(img, 512)
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

def compareGroups(l, group1, group2, same, incomps, outcomps):
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
            #d = compareImagesGivenFeatures(l, group1[i], group2[j])
            d = compareImagesGivenCovariances(l, group1[i], group2[j])
            dist[i,j] = d
            dist[j,i] = d
            avg += d
            count += 1
            if same:
                incomps.append(d)
            else:
                outcomps.append(d)
    avg /= count
    return avg, dist

def compareManyGroups(sn, layer, groupFeatures, incomps, outcomps):
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
            avg, d = compareGroups(layer, groupFeatures[i], groupFeatures[j], i == j, incomps, outcomps)
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

    incomps = np.array(incomps)
    outcomps = np.array(outcomps)
    
    stdin = np.std(incomps)
    stdout = np.std(outcomps)
    return np.mean(incomps), stdin, np.mean(outcomps), stdout
            
def train(trainDir):
    sn = SimpleNet()
    categories = os.listdir(trainDir)
    all_imgs = [map(lambda y:trainDir + "/" + x + "/" + y, os.listdir(trainDir + "/" + x)) for x in categories]

    groupFeatures, groupCovs = computeReprs(sn, all_imgs)
    print "Computed representations"
    layer = "conv5_4"
    incomps = []
    outcomps = []
    meanin, stdin, meanout, stdout = compareManyGroups(sn, layer, groupCovs, incomps, outcomps)
    return meanin, meanout

def test(testDir, meanin, meanout):
    sn = SimpleNet()
    categories = os.listdir(testDir)
    all_imgs = [map(lambda y:testDir + "/" + x + "/" + y, os.listdir(testDir + "/" + x)) for x in categories]

    groupFeatures, groupCovs = computeReprs(sn, all_imgs)
    print "Computed representations"
    layer = "conv5_4"
    incomps = []
    outcomps = []
    compareManyGroups(sn, layer, groupCovs, incomps, outcomps)
    correct, wrong = evaluate(meanin, meanout, incomps, outcomps)
    print float(correct) / (correct + wrong)

def evaluate(meanin, meanout, incomps, outcomps):
    correct = 0
    wrong = 0
    for res in incomps:
        if res - meanin < meanout - res:
            correct += 1
        else:
            wrong += 1
    for res in outcomps:
        if meanout - res < res - meanin:
            correct += 1
        else:
            wrong += 1
    return correct, wrong


if __name__ == "__main__":
    caffe.set_device(int(sys.argv[1]))
    caffe.set_mode_gpu()
    print "Training"
    meanin, meanout = train(sys.argv[2])
    print "Testing"
    test(sys.argv[3], meanin, meanout)

