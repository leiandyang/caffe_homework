import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy_fc8.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = 'images/cat.jpg'

caffe.set_mode_cpu()
# classifier
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
input_image = caffe.io.load_image(IMAGE_FILE)
input_image = input_image
n_iterations = 10000
#n_iterations = 10
label_index = 281  # Index for cat class
caffe_data = np.random.random((1,3,227,227))
caffeLabel = np.zeros((1,1000,1,1))
caffeLabel[0,label_index,0,0] = 1;

print "INPUT IMAGE"
#plt.imshow(input_image)
#plt.show()
print(input_image.shape)

rho = 1;

#print(caffe_data.shape)  # :1,3,227,227
#print(caffeLabel.shape)  # :1,1000,1,1

def visSquare(data1, padsize=1, padval=0):
    data = copy.deepcopy(data1)
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show(block=False)
    return data


for i in range(n_iterations):
    if i%100 == 0:
        print i
    # Perform Forward pass with initial random image (we need to find IMG to max cost function)
    fp_result = net.forward(data=caffe_data)
    # Perform Backward pass
    bp_result = net.backward(**{net.outputs[0]: caffeLabel.reshape(1,1,1,1000)})
    diff = bp_result['data']
    # Perform Gradient ascent and update caffe_data
    caffe_data -= rho * caffe_data * diff
    # Visualize the caffe_data using visSquare function
vis = visSquare(caffe_data.transpose(0,2,3,1))
plt.imshow(vis)
plt.savefig('1st_ex.png')
plt.pause(1)





