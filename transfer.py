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

VGG19_WEIGHTS = {"content":{
                               "conv4_1": 1
},
                     "style": {
                               "conv3_1": 1
                                }}

GOOGLENET_WEIGHTS = {"content":{
                                 "conv2/3x3": 1},
                     "style": {"conv1/7x7_s2": 0.2,
                               "conv2/3x3": 0.2,
                               "inception_3a/output": 0.2,
                               "inception_4a/output": 0.2,
                               "inception_5a/output": 0.2}}

STYLE_SCALE = 1

def transferStyleComplex(styleFile, contentFile, n_iter, ratio):
    styleSpecs, styleContribs = readLayerSpecs(styleFile)
    contentSpecs, contentContribs = readLayerSpecs(contentFile)
    contribs = {"style":styleContribs, "content":contentContribs}

    ns = NeuralStyle()

    ns.transferStyle(styleSpecs, contentSpecs, contribs, n_iter=n_iter, ratio=ratio, length=600)

    img_out = ns.get_generated()
    name = "out" + str(int(time.time())) + ".jpg"
    imsave(name, img_as_ubyte(img_out))

def readLayerSpecs(specsFile):
    """
    A specifications file consists of n lines, each corresponding to one layer.
    Each line specifies the name of the layer, its contribution,
    and a variable number of image file names corresponding to the style or
    content (all separated by spaces).

    @return a dict of the form {layer: [alpha, [image names]], ...]
    """

    out = {}
    outContribs = {}
    with open(specsFile, "r") as specs:
        text = specs.read()
        lines = text.split("\n")
        lines = lines[:len(lines) - 1]
        for ln in lines:
            raw = ln.split(" ")
            out[raw[0]] = raw[2:]
            outContribs[raw[0]] = float(raw[1])
    return out, outContribs

def style_optfn(x, net, weights, layers, reprs, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().

        :param numpy.ndarray x:
            Flattened data array.

        :param caffe.Net net:
            Network to use to generate gradients.

        :param dict weights:
            Weights to use in the network.

        :param list layers:
            Layers to use in the network.

        :param tuple reprs:
            Representation matrices packed in a tuple.

        :param float ratio:
            Style-to-content ratio.
    """

    #update params
    layers_style = weights["style"].keys()
    layers_content = weights["content"].keys()
    net_in = x.reshape(net.blobs["data"].data.shape[1:])

    #compute representations
    (G_style, F_content) = reprs
    (G, F) = _compute_reprs(net_in, net, layers_style, layers_content)

    #backprop by layer
    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    for i, layer in enumerate(reversed(layers)):
        next_layer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]

        #style contribution
        if layer in layers_style:
            wl = weights["style"][layer]
            (l, g) = _compute_style_grad(F, G, G_style, layer)
            loss += wl * l * ratio
            grad += wl * g.reshape(grad.shape) * ratio

        #content contribution
        if layer in layers_content:
            wl = weights["content"][layer]
            (l, g) = _compute_content_grad(F, F_content, layer)
            loss += wl * l
            grad += wl * g.reshape(grad.shape)

        #compute gradient
        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    #format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)

    return loss, grad

def optimizeImage(img, net, contribs, reprs, ratio):
    """
    Optimization function for creating the resulting image.

    """

    contrStyle = contribs["style"]
    contrContent = contribs["content"]
    layersStyle = contrStyle.keys()
    layersContent = contrContent.keys()

    net_in = img.reshape(net.blobs["data"].data.shape[1:])

    (G_guide, F_guide) = reprs
    (G, F) = _compute_reprs(net_in, net, layersStyle, layersContent)

    #backprop by layer
    loss = 0
    layers = list(net.blobs)[1:]
    net.blobs[layers[-1]].diff[:] = 0

    for i, layer in enumerate(reversed(layers)):
        nextLayer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]

        #style contribution
        if layer in layersStyle:
            contr = contrStyle[layer]
            (localLoss, localGrad) = _compute_style_grad(F, G, G_guide, layer)
            loss += contr * localLoss * ratio
            grad += contr * localGrad.reshape(grad.shape) * ratio

        #content contribution
        if layer in layersContent:
            contr = contrContent[layer]
            (localLoss, localGrad) = _compute_content_grad(F, F_guide, layer)
            loss += contr * localLoss
            grad += contr * localGrad.reshape(grad.shape)

        #compute gradient
        net.backward(start=layer, end=nextLayer)
        if nextLayer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[nextLayer].diff[0]

    #format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)

    return loss, grad

def _compute_style_grad(F, G, G_guide, layer):
    """Computes style gradient and loss from activation features."""
    (Fl, Gl) = (F[layer], G[layer])
    c = Fl.shape[0]**-2 * Fl.shape[1]**-2
    El = Gl - G_guide[layer]
    loss = c/4 * (El**2).sum()
    grad = c * sgemm(1.0, El, Fl) * (Fl>0)
    return loss, grad

def _compute_content_grad(F, F_guide, layer):
    """Computes content gradient and loss from activation features."""
    Fl = F[layer]
    El = Fl - F_guide[layer]
    loss = (El**2).sum() / 2
    grad = El * (Fl>0)
    return loss, grad

def _compute_reprs(net_input, net, layers_style, layers_content, gram_scale=1):
    """Computes representation matrices for an image."""
    (repr_style, repr_content) = ({}, {})
    net.blobs["data"].data[0] = net_input
    net.forward()

    for layer in set(layers_style)|set(layers_content):
        F = net.blobs[layer].data[0].copy()
        F.shape = (F.shape[0], -1)
        repr_content[layer] = F
        if layer in layers_style:
            repr_style[layer] = sgemm(gram_scale, F, F.T)
    return repr_style, repr_content

class NeuralStyle:

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

        def callback(xk):
            if self._callback is not None:
                net_in = xk.reshape(self.net.blobs["data"].data.shape[1:])
                self._callback(self.transformer.deprocess("data", net_in))
        self.callback = callback

    def init_weights(self, weights):
        self.weights = weights.copy()

        self.layers = []

        for layer in self.net.blobs:
            if layer in self.weights["content"] or layer in self.weights["style"]:
                self.layers.append(layer)


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

    def get_generated(self):
        data = self.net.blobs["data"].data
        img_out = self.transformer.deprocess("data", data)
        return img_out

    def compReprAllImgs(self, specsStyle, specsContent, length):

        #assume the convnet input is a square
        orig_dim = min(self.net.blobs["data"].shape[2:])
        imgsStyle = {}
        imgsContent = {}

        #create a map from images to layers they appear in
        for layer in specsStyle:
            for img in specsStyle[layer]:
                if img not in imgsStyle:
                    imgsStyle[img] = []
                imgsStyle[img].append(layer)
        for layer in specsContent:
            for img in specsContent[layer]:
                if img not in imgsContent:
                    imgsContent[img] = []
                imgsContent[img].append(layer)

        reprStyle = {}
        reprContent = {}

        for name in imgsStyle:
            img = caffe.io.load_image(name)
            scale = max(length / float(max(img.shape[:2])),
                    orig_dim / float(min(img.shape[:2])))
            img = rescale(img, STYLE_SCALE * scale)
            self._rescale_net(img)
            net_in = self.transformer.preprocess("data", img)
            style, _ = _compute_reprs(net_in, self.net, imgsStyle[name], [])
            for layer in style:
                if layer not in reprStyle:
                    #representation and count (for normalization)
                    reprStyle[layer] = [style[layer], 1]
                else:
                    reprStyle[layer][0] += style[layer]
                    reprStyle[layer][1] += 1
        for layer in reprStyle:
            reprStyle[layer] = reprStyle[layer][0] / reprStyle[layer][1]

        if len(imgsContent) > 0:
            #resize everything to match the first image
            resizeTo = imgsContent.keys()[0]
            dimContent = caffe.io.load_image(resizeTo).shape
        for name in imgsContent:
            img = caffe.io.load_image(name)
            img = resize(img, dimContent)
            scale = max(length / float(max(img.shape[:2])),
                    orig_dim / float(min(img.shape[:2])))
            img = rescale(img, scale)
            self._rescale_net(img)
            net_in = self.transformer.preprocess("data", img)
            _, content = _compute_reprs(net_in, self.net, [], imgsContent[name])
            for layer in content:
                if layer not in reprContent:
                    #representation and count (for normalization)
                    reprContent[layer] = [content[layer], 1]
                else:
                    reprContent[layer][0] += content[layer]
                    reprContent[layer][1] += 1
        for layer in reprContent:
            reprContent[layer] = reprContent[layer][0] / reprContent[layer][1]

        return reprStyle, reprContent

    def transfer_style(self, style_img, content_img, length=512, ratio=1e3,
                       n_iter=512, init="-1", verbose=False, callback=None):
        """Tranform style of the style image onto the content image.
        @param style_img image to get the style from
        @param content_img image to get the content from
        """

        #assume the convnet input is a square
        orig_dim = min(self.net.blobs["data"].shape[2:])
    
        #rescale the style and content images
        scale = max(length / float(max(content_img.shape[:2])),
                    orig_dim / float(min(content_img.shape[:2])))
        content_img = rescale(content_img, scale)

        if type(style_img) is list:
            G_style = self.compute_repr_multiple(style_img)
        else:
            scale = max(length / float(max(style_img.shape[:2])),
                    orig_dim / float(min(style_img.shape[:2])))
            style_img = rescale(style_img, STYLE_SCALE * scale)
        
            #compute style representations
            self._rescale_net(style_img)
            layers = self.weights["style"].keys()
            net_in = self.transformer.preprocess("data", style_img)
            gram_scale = float(content_img.size) / style_img.size
            G_style = _compute_reprs(net_in, self.net, layers, [],
                                    gram_scale=1)[0]
        
        self._rescale_net(content_img)

        layers = self.weights["content"].keys()
        net_in = self.transformer.preprocess("data", content_img)
        F_content = _compute_reprs(net_in, self.net, [], layers)[1]


       # generate initial net input
        # "content" = content image, see kaishengtai/neuralart
        if isinstance(init, np.ndarray):
            img0 = self.transformer.preprocess("data", init)
        elif init == "content":
            img0 = self.transformer.preprocess("data", content_img)
        elif init == "mixed":
            img0 = 0.95*self.transformer.preprocess("data", content_img) + \
                   0.05*self.transformer.preprocess("data", style_img)
        else:
            img0 = self._make_noise_input(init)

        # compute data bounds
        data_min = -self.transformer.mean["data"][:,0,0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])]*(img0.size/3) + \
                      [(data_min[1], data_max[1])]*(img0.size/3) + \
                      [(data_min[2], data_max[2])]*(img0.size/3)

        # optimization params
        grad_method = "L-BFGS-B"
        reprs = (G_style, F_content)
        minfn_args = {
            "args": (self.net, self.weights, self.layers, reprs, ratio),
            "method": grad_method, "jac": True, "bounds": data_bounds,
            "options": {"maxcor": 8, "maxiter": n_iter, "disp": verbose}
        }

        #optimize
        self._callback = callback
        minfn_args["callback"] = self.callback
        n_iters = minimize(style_optfn, img0.flatten(), **minfn_args).nit

    def transferStyle(self, styleSpecs, contentSpecs, contribs,
                      init="-1", n_iter=512, ratio=1e4, length=512, callback=None):

        reprStyle, reprContent = self.compReprAllImgs(styleSpecs, contentSpecs, length)

        if isinstance(init, np.ndarray):
            img0 = self.transformer.preprocess("data", init)
        elif init == "content":
            img0 = self.transformer.preprocess("data", content_img)
        elif init == "mixed":
            img0 = 0.95*self.transformer.preprocess("data", content_img) + \
                   0.05*self.transformer.preprocess("data", style_img)
        else:
            img0 = self._make_noise_input(init)

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
            "args": (self.net, contribs, reprs, ratio),
            "method": optMethod, "jac": True, "bounds": data_bounds,
            "options": {"maxcor": 8, "maxiter": n_iter, "disp": False}
        }

        #optimize
        self._callback = callback
        minfn_args["callback"] = self.callback
        n_iters = minimize(optimizeImage, img0.flatten(), **minfn_args).nit

        


def simple_transfer(n_iter, ratio):
    style_nm = sys.argv[2]
    style_img = caffe.io.load_image(style_nm)
    style = style_nm.split(".")[0]
    content_img = caffe.io.load_image(sys.argv[3])

    t = time.time()
    ns = NeuralStyle(model="vgg19")
    ns.init_weights(VGG19_WEIGHTS)
    ns.transfer_style(style_img, content_img, n_iter=n_iter, init="-1", ratio=ratio)
    img_out = ns.get_generated()
    name = "out" + str(int(time.time())) + ".jpg"
    imsave(name, img_as_ubyte(img_out))

def multiple_transfer(n_iter, ratio):
    content_img = caffe.io.load_image(sys.argv[2])
    if sys.argv[3] == "-d":
        if sys.argv[4][-1] == '/':
            prefix = sys.argv[4]
        else:
            prefix = sys.argv[4] + "/"
        style_imgs = map(lambda f: prefix + f, os.listdir(sys.argv[4]))
    else:
        style_imgs = sys.argv[3:]
    ns = NeuralStyle(model="vgg19")
    ns.init_weights(VGG19_WEIGHTS)
    ns.transfer_style(style_imgs, content_img, n_iter=n_iter, init="-1", ratio=ratio, length=600)
    img_out = ns.get_generated()
    name = "out" + str(int(time.time())) + ".jpg"
    imsave(name, img_as_ubyte(img_out))


def main():
    if sys.argv[1] == "-1":
        caffe.set_mode_cpu()
    else:
        gpu = int(sys.argv[1])
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    n_iter = 400
    ratio = 1e3
    t = time.time()
    #simple_transfer(n_iter, ratio)
    styleFile = sys.argv[2]
    contentFile = sys.argv[3]
    transferStyleComplex(styleFile, contentFile, n_iter, ratio)


if __name__ == "__main__":
    main()


