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
import os

import tensorflow as tf

from cnn_utils import conv, fc, max_pool, avg_pool, batch_norm_layer, load_dict_from_hdf5


class ResNet50(object):

    transfer_layer_flattened_sizes = [227*227*3, 14 * 14 * 1024, 7 * 7 * 2048, 7 * 7 * 2048, 7 * 7 * 2048, 1000]
    transfer_layers_shapes = [(227, 227, 3), (14, 14, 1024), (7, 7, 2048), (7, 7, 2048), (7, 7, 2048), (1, 1, 1000)]

    def __init__(self, model_input, input_layer_name='image', model_name='resnet50', retrain_layers=False,
                 weights_path='DEFAULT'):
        self.model_input = model_input
        self.input_layer_name = input_layer_name
        self.model_name = model_name
        self.retrain_layers = retrain_layers

        if weights_path == 'DEFAULT':
            this_dir, _ = os.path.split(__file__)
            self.weights_path = os.path.join(this_dir, "resources", "resnet50_weights.h5")
        else:
            self.weights_path = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.__create()

    def __create(self):
        with tf.variable_scope(self.model_name):

            self.weights_data = self.__get_weights_data()

            if self.input_layer_name == 'image':
                self.image = tf.reshape(tf.cast(self.model_input, tf.float32), [-1, 227, 227, 3])
                self.__preprocess_image()
                self.__calc_conv1()
                self.__calc_conv2()
                self.__calc_conv3()
                self.__calc_conv4()
                self.__calc_conv5()
                self.__calc_fc6()
                self.transfer_layers = [self.conv1, self.conv2_3, self.conv3_4, self.conv4_6,
                                        self.conv5_1, self.conv5_2, self.conv5_3, self.fc6]
            elif self.input_layer_name == 'conv4_6':
                self.conv4_6 = tf.reshape(tf.cast(self.model_input, tf.float32), [-1, 14, 14, 1024])
                self.__calc_conv5()
                self.__calc_fc6()
                self.transfer_layers = [self.conv5_1, self.conv5_2, self.conv5_3, self.fc6]
            elif self.input_layer_name == 'conv5_1':
                self.conv5_1 = tf.reshape(tf.cast(self.model_input, tf.float32), [-1, 7, 7, 2048])
                self.conv5_2 = tf.nn.relu(
                    self.__identity_block(input_layer=self.conv5_1, name='5b', data=self.weights_data,
                                          num_filters=512))
                self.conv5_3 = tf.nn.relu(
                    self.__identity_block(input_layer=self.conv5_2, name='5c', data=self.weights_data,
                                          num_filters=512))
                self.__calc_fc6()
                self.transfer_layers = [self.conv5_2, self.conv5_3, self.fc6]
            elif self.input_layer_name == 'conv5_2':
                self.conv5_2 = tf.reshape(tf.cast(self.model_input, tf.float32), [-1, 7, 7, 2048])
                self.conv5_3 = tf.nn.relu(
                    self.__identity_block(input_layer=self.conv5_2, name='5c', data=self.weights_data,
                                          num_filters=512))
                self.__calc_fc6()
                self.transfer_layers = [self.conv5_3, self.fc6]
            elif self.input_layer_name == 'conv5_3':
                self.conv5_3 = tf.reshape(tf.cast(self.model_input, tf.float32), [-1, 7, 7, 2048])
                self.__calc_fc6()
                self.transfer_layers = [self.fc6]
            else:
                raise Exception('invalid input layer name... ' + self.input_layer_name)

            self.probs = tf.nn.softmax(self.fc6, name='softmax')


    def __preprocess_image(self):
        # zero-mean input
        with tf.variable_scope('preprocess'):
            mean = tf.constant([104., 117., 124.], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.preprocessed_image = self.image - mean
            # 'RGB'->'BGR'
            self.preprocessed_image = self.preprocessed_image[:, :, :, ::-1]

    def __calc_conv1(self):
        temp = conv(self.preprocessed_image, 7, 7, 64, 2, 2, padding='VALID', name='conv1',
                    data=self.weights_data,
                    retrain_layers=self.retrain_layers)
        temp = batch_norm_layer(temp, 'bn_conv1', self.weights_data)
        self.conv1 = tf.nn.relu(temp)
        self.pool1 = max_pool(self.conv1, 3, 3, 2, 2, padding='SAME', name='pool1')

    def __calc_conv2(self):
        self.conv2_1 = tf.nn.relu(self.__conv_block(input_layer=self.pool1, name='2a', data=self.weights_data,
                                                    num_filters=64, stride_x=1, stride_y=1))
        self.conv2_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv2_1, name='2b', data=self.weights_data,
                                                        num_filters=64))
        self.conv2_3 = tf.nn.relu(self.__identity_block(input_layer=self.conv2_2, name='2c', data=self.weights_data,
                                                        num_filters=64))


    def __calc_conv3(self):
        self.conv3_1 = tf.nn.relu(self.__conv_block(input_layer=self.conv2_3, name='3a', data=self.weights_data,
                                                    num_filters=128))
        self.conv3_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv3_1, name='3b', data=self.weights_data,
                                                        num_filters=128))
        self.conv3_3 = tf.nn.relu(self.__identity_block(input_layer=self.conv3_2, name='3c', data=self.weights_data,
                                                        num_filters=128))
        self.conv3_4 = tf.nn.relu(self.__identity_block(input_layer=self.conv3_3, name='3d', data=self.weights_data,
                                                        num_filters=128))


    def __calc_conv4(self):
        self.conv4_1 = tf.nn.relu(self.__conv_block(input_layer=self.conv3_4, name='4a', data=self.weights_data,
                                                    num_filters=256))
        self.conv4_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv4_1, name='4b', data=self.weights_data,
                                                        num_filters=256))
        self.conv4_3 = tf.nn.relu(self.__identity_block(input_layer=self.conv4_2, name='4c', data=self.weights_data,
                                                        num_filters=256))
        self.conv4_4 = tf.nn.relu(self.__identity_block(input_layer=self.conv4_3, name='4d', data=self.weights_data,
                                                        num_filters=256))
        self.conv4_5 = tf.nn.relu(self.__identity_block(input_layer=self.conv4_4, name='4e', data=self.weights_data,
                                                        num_filters=256))
        self.conv4_6 = tf.nn.relu(self.__identity_block(input_layer=self.conv4_5, name='4f', data=self.weights_data,
                                                        num_filters=256))

    def __calc_conv5(self):
        self.conv5_1 = tf.nn.relu(self.__conv_block(input_layer=self.conv4_6, name='5a', data=self.weights_data,
                                                    num_filters=512))
        self.conv5_2 = tf.nn.relu(self.__identity_block(input_layer=self.conv5_1, name='5b', data=self.weights_data,
                                                        num_filters=512))
        self.conv5_3 = tf.nn.relu(self.__identity_block(input_layer=self.conv5_2, name='5c', data=self.weights_data,
                                                        num_filters=512))

    def __calc_fc6(self):
        # 6th Layer: Flatten -> FC (w ReLu)
        self.pool_6 = avg_pool(self.conv5_3, 7, 7, 7, 7, padding='SAME', name='pool6')
        flattened = tf.reshape(self.pool_6, [-1, 1 * 1 * 2048])
        self.fc6 = fc(flattened, 1 * 1 * 2048, 1000, name='fc1000',
                      data=self.weights_data,
                      retrain_layers=self.retrain_layers)

    def __conv_block(self, input_layer, name, data, num_filters, stride_x=2, stride_y=2):

        with tf.name_scope('conv_block'):
            x = conv(input_layer, 1, 1, num_filters, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2a', data=self.weights_data, retrain_layers=self.retrain_layers)
            x = batch_norm_layer(x, data=data, name='bn' + name + '_branch2a')
            x = tf.nn.relu(x)

            x = conv(x, 3, 3, num_filters, stride_x, stride_y, padding='SAME', name='res' + name +
                                                '_branch2b', data=self.weights_data, retrain_layers=self.retrain_layers)
            x = batch_norm_layer(x, data=data, name='bn' + name + '_branch2b')
            x = tf.nn.relu(x)

            x = conv(x, 1, 1, num_filters*4, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2c', data=self.weights_data, retrain_layers=self.retrain_layers)
            x = batch_norm_layer(x, data=data, name='bn' + name + '_branch2c')

            shortcut = conv(input_layer, 1, 1, num_filters*4, stride_x, stride_y, padding='SAME', name='res' + name +
                                                '_branch1', data=self.weights_data, retrain_layers=self.retrain_layers)
            shortcut = batch_norm_layer(shortcut, data=data, name='bn' + name + '_branch1')

            x = tf.add(x, shortcut)

        return x

    def __identity_block(self, input_layer, name, data, num_filters):

        with tf.name_scope('identity_block'):
            x = conv(input_layer, 1, 1, num_filters, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2a', data=self.weights_data, retrain_layers=self.retrain_layers)
            x = batch_norm_layer(x, data=data, name='bn' + name + '_branch2a')
            x = tf.nn.relu(x)

            x = conv(x, 3, 3, num_filters, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2b', data=self.weights_data, retrain_layers=self.retrain_layers)
            x = batch_norm_layer(x, data=data, name='bn' + name + '_branch2b')
            x = tf.nn.relu(x)

            x = conv(x, 1, 1, num_filters*4, 1, 1, padding='SAME', name='res' + name +
                                                '_branch2c', data=self.weights_data, retrain_layers=self.retrain_layers)
            x = batch_norm_layer(x, data=data, name='bn' + name + '_branch2c')

            x = tf.add(x, input_layer)

        return x

    def __get_weights_data(self):
        # Load the weights and biases into memory
        return load_dict_from_hdf5(self.weights_path)

    def load_initial_weights(self, session):
        if not self.retrain_layers:
            raise Exception('retraining convnet is not enabled')

        # Load the weights into memory
        weights_dict = self.__get_weights_data()

        # Loop over all layer names stored in the weights dict
        for k in weights_dict:
            if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name + "/" + k)) != 0:
                with tf.variable_scope(self.model_name + "/" + k, reuse=True):
                    data = weights_dict[k]
                    for var_name in data:
                        # Biases
                        if str(var_name).endswith("_b:0"):
                            var = tf.get_variable('biases')
                            session.run(var.assign(data[var_name]))
                        # Weights
                        elif str(var_name).endswith("_W:0"):
                            var = tf.get_variable('weights')
                            session.run(var.assign(data[var_name]))

    @staticmethod
    def get_transfer_learning_layer_names():
        return ['image','conv4_6', 'conv5_1', 'conv5_2', 'conv5_2', 'fc6']
