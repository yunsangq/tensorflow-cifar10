import numpy as np
from os import path
import cPickle
# from numpngw import write_png
ROOT = path.dirname(path.dirname(path.abspath(__file__)))


def random_crop(image, size):
    if len(image.shape):
        W, H, D = image.shape
        w, h, d = size
    else:
        W, H = image.shape
        w, h = size
    left, top = np.random.randint(W - w + 1), np.random.randint(H - h + 1)
    return image[left:left+w, top:top+h]


def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    return image[offset_width:offset_width+target_width, offset_height:offset_height+target_height]


def per_image_whitening(image):
    return (image - np.mean(image)) / np.std(image)


def unpickle(filename):
    with open(filename, 'rb') as fp:
        return cPickle.load(fp)


def shuffle(images, labels):
    perm = np.arange(len(labels))
    np.random.shuffle(perm)
    return np.asarray(images)[perm], np.asarray(labels)[perm]


def distort(image, is_train=True):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(float)
    if is_train:
        # write_png('origin_image.png', image.astype(np.uint8))
        image = random_crop(image, (24, 24, 3))
        # write_png('crop_image.png', image.astype(np.uint8))
    else:
        image = crop_to_bounding_box(image, 4, 4, 24, 24)
    image = per_image_whitening(image)
    # write_png('whitening_image.png', image.astype(np.uint8))
    return image


def load_cifar10(is_train=True):
    if is_train:
        filenames = [ROOT + "/tensorflow-cifar10/cifar-10/data_batch_%d" % j for j in xrange(1, 6)]
    else:
        filenames = [ROOT + "/tensorflow-cifar10/cifar-10/test_batch"]
    images, labels = [], []
    for filename in filenames:
        cifar10 = unpickle(filename)
        for i in range(len(cifar10["labels"])):
            images.append(distort(cifar10["data"][i], is_train))
        labels += cifar10["labels"]
    return shuffle(images, np.asarray(labels))

