'''
Copyright 2018 Supun Nakandala and Arun Kumar
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import tensorflow as tf
import numpy as np
import h5py


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, data=None, retrain_layers=False):
    # Get the number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        if not retrain_layers:
            weights = tf.constant(data[name][name + "_W:0"], name='weights', shape=[filter_height, filter_width,
                                                                              input_channels / groups, num_filters])
            biases = tf.constant(data[name][name + "_b:0"], name='biases', shape=[num_filters])
        else:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width,
                                                        input_channels / groups, num_filters], trainable=True)
            biases = tf.get_variable('biases', shape=[num_filters], trainable=True)

    if groups == 1:
        conv = convolve(x, weights)

        # In the case of multiple groups, split inputs & weights convolve them
        # separately. (e.g AlexNet)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    return tf.nn.bias_add(conv, biases, name=name)


def fc(x, num_in, num_out, name, data=None, retrain_layers=False):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        if not retrain_layers:
            weights = tf.constant(data[name][name + "_W:0"], name='weights', shape=[num_in, num_out])
            biases = tf.constant(data[name][name + "_b:0"], name='biases', shape=[num_out])
        else:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', shape=[num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        return tf.nn.xw_plus_b(x, weights, biases, name=name)


def batch_norm_layer(x, name, data=None, retrain_layers=False):
    with tf.variable_scope(name) as scope:
        if not retrain_layers:
            mean = tf.constant(data[name][name + '_running_mean:0'])
            std = tf.constant(data[name][name + '_running_std:0'])
            beta = tf.constant(data[name][name + '_beta:0'])
            gamma = tf.constant(data[name][name + '_gamma:0'])
        else:
            mean = tf.Variable(0, tf.float32, trainable=True)
            std = tf.Variable(0, tf.float32, trainable=True)
            beta = tf.Variable(0, tf.float32, trainable=True)
            gamma = tf.Variable(0, tf.float32, trainable=True)
        return tf.nn.batch_normalization(x, mean=mean, variance=std, offset=beta, scale=gamma, variance_epsilon=1e-12,
                                         name='name')


def max_pool(x, filter_height, fileter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, fileter_width, 1],
                          strides=[1, stride_y, stride_x, 1], padding=padding,
                          name=name)


def avg_pool(x, filter_height, fileter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, filter_height, fileter_width, 1],
                          strides=[1, stride_y, stride_x, 1], padding=padding,
                          name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        __recursively_save_dict_contents_to_group(h5file, '/', dic)


def __recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            __recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/')


def __recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = __recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
