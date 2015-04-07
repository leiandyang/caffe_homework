import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
import fractions
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import matplotlib.cm as cm
import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy_fc8.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/cat.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
input_image = caffe.io.load_image(IMAGE_FILE)
input_image = input_image

n_iterations = 10000
label_index = 281  # Index for cat class
caffe_data = np.random.random((1,3,227,227))
caffeLabel = np.zeros((1,1000,1,1))
caffeLabel[0,label_index,0,0] = 1;


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data,cmap=cm.gray)

def factor_partial(N):
    for R in xrange(int(np.sqrt(N)),1,-1):
        if N%R == 0:
            return R

#Perform a forward pass with the data as the input image
pr_result = net.predict([input_image])

#Perform a backward pass for the cat class (281)
bp_result = net.backward(**{net.outputs[0]:caffeLabel.reshape(1,1,1,1000)})
diff = bp_result['data']


# Plot each derivative of each layer and save each fig.

for blobName, v in net.blobs.items():
    print (blobName, v.diff.shape, v.diff.ndim)
    if v.diff.ndim ==4:
        grad = np.mean(np.absolute(v.diff), axis=1)
        # Averaging by # channels or # kernels
        grad = grad[0,:,:]
    if v.diff.ndim ==2:
        grad = np.absolute(v.diff)
        grad = grad.reshape(factor_partial(v.diff.size), (v.diff.size)/factor_partial(v.diff.size))

    print grad.shape
    plt.imshow(grad, cmap=cm.gray_r)
    fout = blobName+'.png'
    plt.savefig(fout)
    plt.pause(1)


#        print grad.shape
#    elif v.data.ndim==2:
#        grad = np.mean
#('data', (1, 3, 227, 227))
#('conv1', (1, 96, 55, 55))
#('pool1', (1, 96, 27, 27))
#('norm1', (1, 96, 27, 27))
#('conv2', (1, 256, 27, 27))
#('pool2', (1, 256, 13, 13))
#('norm2', (1, 256, 13, 13))
#('conv3', (1, 384, 13, 13))
#('conv4', (1, 384, 13, 13))
#('conv5', (1, 256, 13, 13))
#('pool5', (1, 256, 6, 6))
#('fc6', (1, 4096))
#('fc7', (1, 4096))
#('fc8', (1, 1000))

