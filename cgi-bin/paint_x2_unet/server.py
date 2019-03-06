import numpy as np
import argparse
import chainer
from PIL import Image
from chainer import cuda, serializers, Variable  # , optimizers, training
import cv2
import os.path
import time

from mldraw_adaptor import canvas_message_handler, start

from img2imgDataset import ImageAndRefDataset

import unet
import lnet

cnn_128 = unet.UNET()
cnn_512 = unet.UNET()

cnn_128.to_gpu()
cnn_512.to_gpu()

serializers.load_npz(
    "./models/unet_128_standard", cnn_128)

serializers.load_npz(
    "./models/unet_512_standard", cnn_512)

cuda.get_device(0).use()


def cvt2YUV(img):
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor( img, cv2.COLOR_RGB2YUV )
    else:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2YUV )
    return img

def preprocess(lines, colours, blur=0, s_size=128):
    lines = lines[:, :, 0]
    image1 = np.asarray(lines, np.float32)

    _image1 = image1.copy()
    if image1.shape[0] < image1.shape[1]:
        s0 = s_size
        s1 = int(image1.shape[1] * (s_size / image1.shape[0]))
        s1 = s1 - s1 % 16
        _s0 = 4 * s0
        _s1 = int(image1.shape[1] * ( _s0 / image1.shape[0]))
        _s1 = (_s1+8) - (_s1+8) % 16
    else:
        s1 = s_size
        s0 = int(image1.shape[0] * (s_size / image1.shape[1]))
        s0 = s0 - s0 % 16
        _s1 = 4 * s1
        _s0 = int(image1.shape[0] * ( _s1 / image1.shape[1]))
        _s0 = (_s0+8) - (_s0+8) % 16

    _image1 = image1.copy()
    _image1 = cv2.resize(_image1, (_s1, _s0),
                            interpolation=cv2.INTER_AREA)
    #noise = np.random.normal(0,5*np.random.rand(),_image1.shape).astype(self._dtype)

    if blur > 0:
        blured = cv2.blur(_image1, ksize=(blur, blur))
        image1 = _image1 + blured - 255

    image1 = cv2.resize(image1, (s1, s0), interpolation=cv2.INTER_AREA)

    if image1.ndim == 2:
        image1 = image1[:, :, np.newaxis]
    if _image1.ndim == 2:
        _image1 = _image1[:, :, np.newaxis]

    image1 = np.insert(image1, 1, -512, axis=2)
    image1 = np.insert(image1, 2, 128, axis=2)
    image1 = np.insert(image1, 3, 128, axis=2)

    Image.fromarray(image1.clip(0, 255).astype(np.uint8)).save('image1.png')
    image_ref = colours
    image_ref = cv2.resize(image_ref, (image1.shape[1], image1.shape[
                            0]), interpolation=cv2.INTER_NEAREST)

    b, g, r, a = cv2.split(image_ref)
    image_ref = cvt2YUV( cv2.merge((b, g, r)) )

    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            if a[x][y] != 0:
                for ch in range(3):
                    image1[x][y][ch + 1] = image_ref[x][y][ch]

    return image1.transpose(2, 0, 1), _image1.transpose(2, 0, 1)

def colorize(lines, colour, blur=0, s_size=128):
    sample = preprocess(lines, colour, blur, s_size)
    sample_container = cuda.to_gpu(sample[0][np.newaxis])

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            image_conv2d_layer = cnn_128.calc(Variable(sample_container))

    input_bat = np.zeros((1, 4, sample[1].shape[1], sample[1].shape[2]), dtype='f')

    input_bat[0, 0, :] = sample[1]

    output = cuda.to_cpu(image_conv2d_layer.data[0])

    for channel in range(3):
        input_bat[0, 1 + channel, :] = cv2.resize(
            output[channel, :], 
            (sample[1].shape[2], sample[1].shape[1]), 
            interpolation=cv2.INTER_CUBIC)

    link = cuda.to_gpu(input_bat, None)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            image_conv2d_layer = cnn_512.calc(Variable(link))
    array = image_conv2d_layer.data[0]
    array = array.transpose(1, 2, 0)
    array = array.clip(0, 255).astype(np.uint8)
    array = cuda.to_cpu(array)
    (major, minor, _) = cv2.__version__.split(".")
    if major == '3':
        img = cv2.cvtColor(array, cv2.COLOR_YUV2RGB)
    else:
        img = cv2.cvtColor(array, cv2.COLOR_YUV2BGR)
    return img

@canvas_message_handler('colorize')
def colorize(img):
    arr = np.asarray(img)
    # only black
    lines = np.copy(arr)
    lines[lines.sum(2) != 0, :] = 255
    lines[:, :, 3] = 255
    colours = np.copy(arr)
    colours[colours.sum(2) == 0, :] = 0
    # only colours
    result = colorize(lines, colours)
    return Image.fromarray(result)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-url', type=str, help='URL of backend server')
    parser.add_argument('--self-url', type=str, help='URL of this computer')
    return parser.parse_args()

if __name__ == '__main__':
    start(args())