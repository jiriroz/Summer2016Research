import os
import sys

os.environ['GLOG_minloglevel'] = '2' #suppress network init log
import caffe
import math
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

DEFAULTS = {
    "length":512,
    "ratio":1e4,
    "iters":400,
    "styleScale":1,
    "init":"-1",
    "colorless":0,
    "maxClass":-1
}

def transferStyleComplex(inFile):
    specs = readJsonInput(inFile)

    ratio = specs["ratio"]
    iters = specs["iters"]
    init = specs["init"]

    ns = NeuralStyle()

    t = time.time()

    ns.loadParam(specs)
    ns.transferStyle(specs, n_iter=iters, ratio=ratio, init=init)

    img_out = ns.get_generated()
    if len(sys.argv) > 3:
        name = sys.argv[3]
    else:
        name = "out" + str(int(time.time())) + ".jpg"
    imsave(name, img_as_ubyte(img_out))
    print "took " + str(int(time.time() - t)) + " s"

def readJsonInput(jsonFile):
    """
    Parses a specifications file, which is a JSON file, and fills in missing values.
    Parateters of the specs file:
    @style
    @content
    @length
    @ratio
    @iters
    @styleScale

    @return specifications dict where keys are parameters.
    """
    with open(jsonFile, "r") as f:
        raw = f.read()
    specs = json.loads(raw)

    #If contributions for individial images not given, make them uniform
    #If a weight for a layer not given, make it uniform.
    #TODO: Consider normalizing weights/contribs
    nLayers = len(specs["style"])
    for layer in specs["style"]:
        if "weight" not in specs["style"][layer]:
            specs["style"][layer]["weight"] = 1.0 / nLayers
        if "contributions" not in specs["style"][layer]:
            n = len(specs["style"][layer]["images"])
            if n > 0:
                contr = [1.0/n] * n
            else:
                contr = 0
            specs["style"][layer]["contributions"] = contr

    nLayers = len(specs["content"])
    for layer in specs["content"]:
        if "weight" not in specs["content"][layer]:
            specs["content"][layer]["weight"] = 1.0 / nLayers
        if "contributions" not in specs["content"][layer]:
            n = len(specs["content"][layer]["images"])
            if n > 0:
                contr = [1.0/n] * n
            else:
                contr = 0
            specs["content"][layer]["contributions"] = contr

    #load default values if not present
    for value in DEFAULTS:
        if value not in specs:
            specs[value] = DEFAULTS[value]

    return specs

class NeuralStyle:

    def __init__(self, model="vgg19"):

        if model == "vgg19":
            model_file = "models/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt"
            pretrained_file = "models/vgg19/VGG_ILSVRC_19_layers.caffemodel"
            mean = np.array([103.939, 116.779, 123.68])
        else:
            model_file = "models/googlenet/deploy.prototxt"
            pretrained_file = "models/googlenet/googlenet_style.caffemodel"
            mean = "models/googlenet/ilsvrc_2012_mean.npy"

        self.load_model(model_file, pretrained_file, mean)

        self.gradientMask = {}
        for layer in self.net.blobs.keys():
            n = self.net.blobs[layer].data[0].shape[0]
            self.gradientMask[layer] = np.ones((n, n))

        def callback(xk):
            if self._callback is not None:
                net_in = xk.reshape(self.net.blobs["data"].data.shape[1:])
                self._callback(self.transformer.deprocess("data", net_in))
        self.callback = callback

    def loadParam(self, specs):
        self.styleScale = specs["styleScale"]
        self.length = specs["length"]
        self.colorless = specs["colorless"] == 1
        self.maxClass = specs["maxClass"]

    def load_model(self, model_file, pretrained_file, mean):
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

    def transferStyle(self, specs, init="-1", n_iter=512,
                      ratio=1e4, callback=None):
        #assume the convnet input is a square
        origDim = min(self.net.blobs["data"].shape[2:])
        reprStyle, reprContent = self.compReprAllImgs(origDim, specs["style"], specs["content"])
        
        if isinstance(init, np.ndarray):
            img0 = self.transformer.preprocess("data", init)
        elif init == "-1":
            img0 = self._make_noise_input(init)
        else:
            _img = caffe.io.load_image(init)
            img0 = self.transformer.preprocess("data", _img)
        # compute data bounds
        data_min = -self.transformer.mean["data"][:,0,0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])]*(img0.size/3) + \
                      [(data_min[1], data_max[1])]*(img0.size/3) + \
                      [(data_min[2], data_max[2])]*(img0.size/3)

        # optimization params
        optMethod = "L-BFGS-B"
        reprs = (reprStyle, reprContent)
        minfn_args = {
            "args": (specs, reprs, ratio),
            "method": optMethod, "jac": True, "bounds": data_bounds,
            "options": {"maxcor": 8, "maxiter": n_iter, "disp": False}
        }

        #optimize
        self._callback = callback
        minfn_args["callback"] = self.callback
        n_iters = minimize(self.optimizeImage, img0.flatten(), **minfn_args).nit


    def compReprAllImgs(self, origDim, specsStyle, specsContent):
        imgsStyle = {}
        imgsContent = {}

        #create a map from images to layers they appear in
        for layer in specsStyle:
            for i in range(len(specsStyle[layer]["images"])):
                #iterate over every image for that layer
                img = specsStyle[layer]["images"][i]
                contr = specsStyle[layer]["contributions"][i]
                if img not in imgsStyle:
                    imgsStyle[img] = []
                imgsStyle[img].append((layer, contr))
        for layer in specsContent:
            for i in range(len(specsContent[layer]["images"])):
                #iterate over every image for that layer
                img = specsContent[layer]["images"][i]
                contr = specsContent[layer]["contributions"][i]
                if img not in imgsContent:
                    imgsContent[img] = []
                imgsContent[img].append((layer, contr))

        reprStyle = {}
        reprContent = {}

        for name in imgsStyle:
            img = caffe.io.load_image(name)
            scale = max(self.length / float(max(img.shape[:2])),
                    origDim / float(min(img.shape[:2])))
            img = rescale(img, self.styleScale * scale)
            self._rescale_net(img)
            net_in = self.transformer.preprocess("data", img)
            layersStyle = [x[0] for x in imgsStyle[name]]
            style, _ = self._compute_reprs(net_in, layersStyle, [])
            for (layer, contr) in imgsStyle[name]:
                if layer not in reprStyle:
                    reprStyle[layer] = contr * style[layer]
                else:
                    reprStyle[layer] += contr * style[layer]

        if len(imgsContent) > 0:
            #resize everything to match the first image
            #this will also be the size of the output image
            resizeTo = imgsContent.keys()[0]
            contentMain = caffe.io.load_image(resizeTo)
            scale = max(self.length / float(max(contentMain.shape[:2])),
                    origDim / float(min(contentMain.shape[:2])))
            contentMain = rescale(contentMain, scale)
            dimContent = contentMain.shape
            #save this image as the main content image (for colorless transfer)
            self.contentImg = contentMain
        for name in imgsContent:
            img = caffe.io.load_image(name)
            img = resize(img, dimContent)
            scale = max(self.length / float(max(img.shape[:2])),
                    origDim / float(min(img.shape[:2])))
            img = rescale(img, scale)
            self._rescale_net(img)
            net_in = self.transformer.preprocess("data", img)
            layersContent = [x[0] for x in imgsContent[name]]
            _, content = self._compute_reprs(net_in, [], layersContent)
            for (layer, contr) in imgsContent[name]:
                if layer not in reprContent:
                    reprContent[layer] = contr * content[layer]
                else:
                    reprContent[layer] += contr * content[layer]

        return reprStyle, reprContent

    def input_image(self, img):
        self._rescale_net(img)
        net_in = self.transformer.preprocess("data", img)
        self.net.blobs["data"].data[0] = net_in

    def _rescale_net(self, img):
        new_dims = (1, img.shape[2]) + img.shape[:2]
        self.net.blobs["data"].reshape(*new_dims)
        self.transformer.inputs["data"] = new_dims

    def _make_noise_input(self, init):
        """
            Creates an initial input (generated) image.
        """

        # specify dimensions and create grid in Fourier domain
        dims = tuple(self.net.blobs["data"].data.shape[2:]) + \
               (self.net.blobs["data"].data.shape[1], )
        grid = np.mgrid[0:dims[0], 0:dims[1]]

        # create frequency representation for pink noise
        Sf = (grid[0] - (dims[0]-1)/2.0) ** 2 + \
             (grid[1] - (dims[1]-1)/2.0) ** 2
        Sf[np.where(Sf == 0)] = 1
        Sf = np.sqrt(Sf)
        Sf = np.dstack((Sf**int(init),)*dims[2])

        # apply ifft to create pink noise and normalize
        ifft_kernel = np.cos(2*np.pi*np.random.randn(*dims)) + \
                      1j*np.sin(2*np.pi*np.random.randn(*dims))
        img_noise = np.abs(ifftn(Sf * ifft_kernel))
        img_noise -= img_noise.min()
        img_noise /= img_noise.max()

        # preprocess the pink noise image
        x0 = self.transformer.preprocess("data", img_noise)

        return x0

    def optimizeImage(self, img, contribs, reprs, ratio):
        """
        Optimization function for creating the resulting image.
        """

        contrStyle = contribs["style"]
        contrContent = contribs["content"]
        layersStyle = contrStyle.keys()
        layersContent = contrContent.keys()

        net_in = img.reshape(self.net.blobs["data"].data.shape[1:])

        (G_guide, F_guide) = reprs
        (G, F) = self._compute_reprs(net_in, layersStyle, layersContent)

        #backprop by layer
        loss = 0
        layers = list(self.net.blobs)[1:]
        for i, layer in enumerate(reversed(layers)):
            nextLayer = None if i == len(layers)-1 else layers[-i-2]
            grad = self.net.blobs[layer].diff[0]

            if layer == "prob" and self.maxClass != -1:
                (localLoss, localGrad) = self.crossEntropyLoss()
                loss += localLoss * 1.0
                grad += localGrad * 1.0

            #style contribution
            if layer in layersStyle:
                contr = contrStyle[layer]["weight"]
                (localLoss, localGrad) = self._compute_style_grad(F, G, G_guide, layer)
                loss += contr * localLoss * ratio
                grad += contr * localGrad.reshape(grad.shape) * ratio

            #content contribution
            if layer in layersContent:
                contr = contrContent[layer]["weight"]
                (localLoss, localGrad) = self._compute_content_grad(F, F_guide, layer)
                loss += contr * localLoss
                grad += contr * localGrad.reshape(grad.shape)

            #compute gradient
            self.net.backward(start=layer, end=nextLayer)
            if nextLayer is None:
                grad = self.net.blobs["data"].diff[0]
            else:
                grad = self.net.blobs[nextLayer].diff[0]

        #format gradient for minimize() function
        grad = grad.flatten().astype(np.float64)

        return loss, grad

    def crossEntropyLoss(self):
        p = self.net.blobs["prob"].data[0]
        fc8 = self.net.blobs["fc8"].data[0]
        k = self.maxClass
        loss = -math.log(p[k])
        grad = - np.e ** (fc8 + fc8[k]) / np.sum(fc8)
        grad[k] = p[k] * (1 - p[k])
        return loss, grad

    def _compute_reprs(self, net_input, layersStyle, layersContent, gram_scale=1):
        """Computes representation matrices for an image."""

        (repr_style, repr_content) = ({}, {})
        self.net.blobs["data"].data[0] = net_input
        self.net.forward()

        for layer in set(layersStyle)|set(layersContent):
            F = self.net.blobs[layer].data[0].copy()
            F.shape = (F.shape[0], -1)
            repr_content[layer] = F
            if layer in layersStyle:
                repr_style[layer] = sgemm(gram_scale, F, F.T)
        return repr_style, repr_content

    def compute_style_grad(self, F, G, G_guide, layer):
        """Computes style gradient and loss from activation features."""
        (Fl, Gl) = (F[layer], G[layer])
        c = Fl.shape[0]**-1 * Fl.shape[1]**-1 #modification
        El = (Gl - G_guide[layer]) * self.gradientMask[layer]
        loss = c/2 * (El**2).sum() #modification
        grad = c * sgemm(1.0, El, Fl) * (Fl>0)
        return loss, grad

    def _compute_style_grad(self, F, G, G_guide, layer):
        """Computes style gradient and loss from activation features."""
        (Fl, Gl) = (F[layer], G[layer])
        c = Fl.shape[0]**-2 * Fl.shape[1]**-2
        El = (Gl - G_guide[layer]) * self.gradientMask[layer]
        loss = c/4 * (El**2).sum()
        grad = c * sgemm(1.0, El, Fl) * (Fl>0)
        return loss, grad

    def _compute_content_grad(self, F, F_guide, layer):
        """Computes content gradient and loss from activation features."""
        Fl = F[layer]
        El = Fl - F_guide[layer]
        loss = (El**2).sum() / 2
        grad = El * (Fl>0)
        return loss, grad

    def get_generated(self):
        data = self.net.blobs["data"].data
        img_out = self.transformer.deprocess("data", data)
        if self.colorless:
            img_out = self.originalColors(self.contentImg, img_out)
        return img_out

    def originalColors(self, content, generated):
        content_yuv = self.rgb2yuv(content)
        generated_yuv = self.rgb2yuv(generated)
        generated_yuv[:,:,1] = content_yuv[:,:,1]
        generated_yuv[:,:,2] = content_yuv[:,:,2]
        return np.clip(self.yuv2rgb(generated_yuv), -1, 1)

    def rgb2yuv(self, img):
        img_yuv = np.zeros(img.shape)
        img_yuv[:,:,0] = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        img_yuv[:,:,1] = 0.492*(img[:,:,2] - img_yuv[:,:,0])
        img_yuv[:,:,2] = 0.877*(img[:,:,0] - img_yuv[:,:,0])
        return img_yuv

    def yuv2rgb(self, img):
        img_rgb = np.zeros(img.shape)
        img_rgb[:,:,0] = img[:,:,0] + 1.14*img[:,:,2]
        img_rgb[:,:,1] = img[:,:,0] - 0.395*img[:,:,1] - 0.581*img[:,:,2]
        img_rgb[:,:,2] = img[:,:,0] + 2.033*img[:,:,1]
        return img_rgb


def main():
    if sys.argv[1] == "-1":
        caffe.set_mode_cpu()
    else:
        gpu = int(sys.argv[1])
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    infile = sys.argv[2]
    transferStyleComplex(infile)


if __name__ == "__main__":
    main()


