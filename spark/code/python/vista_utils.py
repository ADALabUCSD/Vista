# coding=utf-8
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

from pyspark import SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructField, StringType, StructType, BinaryType, IntegerType, FloatType
from pyspark.sql.functions import col, array, broadcast, lit
from pyspark.sql.column import _to_java_column, _to_seq, Column
from pyspark.sql import DataFrame

from cnn.alexnet import AlexNet
from cnn.resnet50 import ResNet50
from cnn.vgg16 import VGG16

import tensorflow as tf
import tensorframes as tfs


def get_struct_df(sc, data_file_path):
    """
        Reads the structured data csv file from HDFS and returns a DataFrame.
    :param sc: SparkContext
    :param data_file_path: HDFS csv file path
    :return: DataFrame
    """
    sql_context = SQLContext(sc)
    struct_df = sql_context.read.format('csv').options(header='false').load(data_file_path)
    col_names = struct_df.schema.names
    struct_df = struct_df.withColumn("id", struct_df[col_names[0]].cast(StringType())) \
        .withColumn("features", array([col(x).cast(FloatType()) for x in col_names[1:-1]])) \
        .withColumn("label", struct_df[col_names[-1]].cast(IntegerType())) \
        .select("id", "features", "label")
    return struct_df


def get_images_df(sc, image_dir_path):
    """
        Reads images from HDFS and returns a DataFrame.
    :param sc: SparkContext
    :param image_dir_path: HDFS image dir. path
    :return: DataFrame
    """
    sql_context = SQLContext(sc)
    return DataFrame(sc._jvm.vista.udf.VistaUDFs.getImagesDF(sc._jsc, image_dir_path), sql_context)


def downstream_ml_func(features_df, results_dict, layer_index):
    """
        Sample implementation fo the downstream ML function
    :param features_df: Merged (struct+cnn) feature DataFrame
    :param results_dict: Dictionary object which is used to store downstream ML model performance details such as accuracy.
    :param layer_index: Layer index of the CNN of which the current features_df correspond to. The layer index is negative
                        pointing the index from the top of the CNN layers
    :return: Dictionary
    """
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.5)
    model = lr.fit(features_df)
    predictions = model.transform(features_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    results_dict[layer_index] = evaluator.evaluate(predictions)
    return results_dict


def slice_layers_udf(sc, image_features, cum_sums):
    """
        Slice bulk inference cnn features into multiple layers
    :param sc: SparkContext
    :param image_features: Bulk inference CNN image features
    :param cum_sums: List containing cumulative sizes of the CNN layer feature sizes
    :return: DataFrame
    """
    _slice_layers = sc._jvm.vista.udf.VistaUDFs.sliceLayersUDF()
    return Column(_slice_layers.apply(_to_seq(sc, [image_features, cum_sums], _to_java_column)))


def image_to_byte_arr_udf(sc, image_buffer):
    """
        Transforms the JPEG encoded images to raw format returns a DataFrame of byte[]
    :param sc: SparkContext
    :param image_buffer: Images in JPEG format
    :return: DataFrame
    """
    _image_to_byte_arr = sc._jvm.vista.udf.VistaUDFs.imageToByteArrayUDF()
    return Column(_image_to_byte_arr.apply(_to_seq(sc, [image_buffer], _to_java_column)))


def merge_features_udf(sc, layer, x, y, z, image_features, structured_features):
    """
        Merge structured and cnn features into one array
    :param sc:
    :param layer:
    :param x:
    :param y:
    :param z:
    :param image_features:
    :param structured_features:
    :return:
    """
    _merge_features = sc._jvm.vista.udf.VistaUDFs.mergeFeaturesUDF()
    return Column(
        _merge_features.apply(_to_seq(sc, [layer, image_features, structured_features, x, y, z], _to_java_column)))


def serialize_cnn_features_udf(sc, arr):
    """
        Serialize CNN features
    :param sc: SparkContext
    :param arr: CNN features DataFrame
    :return: DataFrame
    """
    _serialize_array = sc._jvm.vista.udf.VistaUDFs.serializeCNNFeaturesArrUDF()
    return Column(_serialize_array.apply(_to_seq(sc, [arr], _to_java_column)))


def get_joined_features(image_features_df, struct_df, broadcast_hash_join):
    """
        Joins the structured DataFrame and cnn features DataFrame
    :param image_features_df: DataFrame containing image features
    :param struct_df: DataFrame containing structured features
    :param broadcast_hash_join: Boolean. Whether to use broadcast join or hash join
    :return: DataFrame
    """
    if broadcast_hash_join:
        features_df = image_features_df.alias('x').join(broadcast(struct_df.alias('y')), col('x.id') == col('y.id'))
    else:
        features_df = image_features_df.alias('x').join(struct_df.alias('y'), col('x.id') == col('y.id'))

    features_df = features_df.select('x.id', 'x.image_features', 'y.features', 'y.label')
    return features_df


def get_feature_projections(sc, features_df, num_layers_to_explore, shapes):
    """
        Projects CNN features for each layer in the bulk CNN inference approach.
    :param sc: SparkContext
    :param features_df: DataFrame of merged features
    :param num_layers_to_explore: number of layers to be explored
    :param shapes: The shapes of CNN feature layers
    :return: DataFrame
    """
    return [features_df.select('label', merge_features_udf(sc, lit(layer), lit(shapes[layer][0]), lit(shapes[layer][1]),
                                                           lit(shapes[layer][2]), col('image_features'),
                                                           col('features'))
                               .alias('features')) for layer in range(num_layers_to_explore)]


def get_all_image_features(model_name, joined_df, num_layers_to_explore, cnn_input_layer_index=0):
    """
        Bulk cnn inference
    :param model_name: CNN model name (AlexNet, VGG16, ResNet50)
    :param joined_df: Input DataFrame containing structured features and raw images
    :param num_layers_to_explore: Number of layer from the top of the CNN to be explored
    :param cnn_input_layer_index: Starting layer index. Zero means raw images
    :return: DataFrame
    """
    g = tf.Graph()
    with g.as_default():
        image_buffer = tf.placeholder(tf.string, [], 'input_layer')
        image = tf.decode_raw(image_buffer, tf.float32)

        if model_name == 'alexnet':
            model = AlexNet(image, input_layer_name=AlexNet.get_transfer_learning_layer_names()[cnn_input_layer_index],
                            model_name='alexnet')
        elif model_name == 'resnet50':
            model = ResNet50(image,
                             input_layer_name=ResNet50.get_transfer_learning_layer_names()[cnn_input_layer_index],
                             model_name='resnet50')
        elif model_name == 'vgg16':
            model = VGG16(image, input_layer_name=VGG16.get_transfer_learning_layer_names()[cnn_input_layer_index],
                          model_name='vgg16')

        concat_layers = [tf.reshape(model.transfer_layers[-1 * i], [-1, model.transfer_layer_flattened_sizes[-1 * i]])
                         for i in range(1, num_layers_to_explore + 1)]
        output = tf.concat(concat_layers, 1, name='image_features')

        cumulative_sizes = [0]
        for i in range(1, num_layers_to_explore + 1):
            cumulative_sizes.append(cumulative_sizes[i - 1] + model.transfer_layer_flattened_sizes[-1 * i])

        image_features_df = tfs.map_rows(output, joined_df)
    return image_features_df, cumulative_sizes, model.transfer_layers_shapes[-1 * num_layers_to_explore:]


def get_image_features_for_layer(model_name, layer_num_from_top, starting_layer_df, starting_layer, joined=True):
    """
        Staged CNN inference.
    :param model_name: CNN model name (AlexNet, VGG16, ResNet50)
    :param layer_num_from_top: Layer index from the top most layer of the CNN
    :param starting_layer_df: Input DataFrame for the staged inference
    :param starting_layer: Starting layer index. Zero means raw images
    :param joined: Boolean. Whether the input DataFrame is already joined with structured features.
    :return: DataFrame
    """
    g = tf.Graph()
    with g.as_default():
        input_buffer = tf.placeholder(tf.string, [], 'input_layer')
        input = tf.decode_raw(input_buffer, tf.float32)

        if model_name == 'alexnet':
            input_layer_name = AlexNet.get_transfer_learning_layer_names()[starting_layer]
            model = AlexNet(input, input_layer_name=input_layer_name, model_name='alexnet')
        elif model_name == 'resnet50':
            input_layer_name = ResNet50.get_transfer_learning_layer_names()[starting_layer]
            model = ResNet50(input, input_layer_name=input_layer_name, model_name='resnet50')
        elif model_name == 'vgg16':
            input_layer_name = VGG16.get_transfer_learning_layer_names()[starting_layer]
            model = VGG16(input, input_layer_name=input_layer_name, model_name='vgg16')

        output = tf.reshape(model.transfer_layers[layer_num_from_top],
                            [-1, model.transfer_layer_flattened_sizes[layer_num_from_top]], name='image_features')

        if joined:
            image_features_df = tfs.map_rows(output, starting_layer_df).select(col('id'), col('image_features'),
                                                                               col('features'), col('label'))
        else:
            image_features_df = tfs.map_rows(output, starting_layer_df).select(col('id'), col('image_features'))

    return image_features_df, model.transfer_layers_shapes[layer_num_from_top]


def get_dir_size(dir_path):
    """
        Read HDFS metadata and estimate the size of image files.
    :param dir_path: Images dir. path on HDFS
    :return: Total size of images in Bytes
    """
    size_in_bytes = 0
    if dir_path.startswith("hdfs://"):
        output = os.popen("hadoop fs -du /" + dir_path.split(":")[2].split("/")[1]).read().split("\n")
        for line in output:
            str = line.split(" ")[0].strip()
            if len(str) != 0:
                size_in_bytes += int(line.split(" ")[0])
    else:#file://
        dir_path = dir_path.replaceAll('file://', '')
        size_in_bytes = sum(os.path.getsize(f) for f in os.listdir(dir_path) if os.path.isfile(f))

    return size_in_bytes
