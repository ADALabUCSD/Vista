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

from __future__ import print_function, division

import sys

from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

sys.path.append('../code/python')

from vista_utils import get_struct_df
from vista_utils import get_joined_features, serialize_cnn_features_udf, downstream_ml_func
from vista_utils import get_image_features_for_layer, get_feature_projections

from cnn.alexnet import AlexNet
from cnn.vgg16 import VGG16
from cnn.resnet50 import ResNet50

# Slight modification to the baseline approach where a base layer is pre-materialized and
# instead of starting the CNN inference all the way from raw images start from the materialized
# base layer.
if __name__ == '__main__':
    ############################change appropriately###################################
    model = 'alexnet'
    pre_mat_layer_index = -4  # from the top
    explore_layer_index = -1  # from the top
    struct_input = 'hdfs://spark-cluster-master:9000/foods.csv'
    pre_mat_input = 'hdfs://spark-cluster-master:9000/' + model + "_pre_mat_layer.parquet"
    heap_memory = 29
    executor_cpu = 5
    storage_level = StorageLevel(True, True, False, True)  # memory and disk deserialized
    sp_core_memory_fraction = 0.6
    ###################################################################################

    if model == 'vgg16':
        initial_shape = VGG16.transfer_layers_shapes[pre_mat_layer_index]
    elif model == 'resnet50':
        initial_shape = ResNet50.transfer_layers_shapes[pre_mat_layer_index]
    elif model == 'alexnet':
        initial_shape = AlexNet.transfer_layers_shapes[pre_mat_layer_index]

    conf = SparkConf()
    app_name = 'pre-mat-' + model + "-l:" + str(pre_mat_layer_index)
    conf.setAppName(app_name)
    conf.set("spark.executor.memory", str(heap_memory)+"g")
    conf.set("spark.memory.fraction", sp_core_memory_fraction)
    conf.set("spark.executor.cores", executor_cpu)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.reduceLocality.enabled", "false")
    conf.set("spark.files.maxPartitionBytes", "10485760")  # 10MB

    sc = SparkContext.getOrCreate(conf=conf)
    sql_context = SQLContext(sc)

    struct_df = get_struct_df(sc, struct_input)

    images_df = sql_context.read.parquet(pre_mat_input)

    if pre_mat_layer_index == explore_layer_index:
        features_df = images_df.alias('x')\
            .join(struct_df.alias('y'), col('x.id') == col('y.id'))\
            .select('x.id', col('x.input_layer').alias('image_features'), 'y.features', 'y.label')

        merged_features_df = get_feature_projections(sc, features_df, 1, [initial_shape])[0]
    else:
        images_df = images_df.select(col('id'), serialize_cnn_features_udf(sc, col('input_layer')).alias('input_layer'))
        image_features_df, shape = get_image_features_for_layer(model, explore_layer_index, images_df, pre_mat_layer_index, False)
        features_df = get_joined_features(image_features_df, struct_df, False)
        features_df = features_df.select("id", "features", "image_features", "label")
        merged_features_df = get_feature_projections(sc, features_df, 1, [shape])[0]

    merged_features_df._jdf.persist(sc._getJavaStorageLevel(storage_level))
    print(downstream_ml_func(merged_features_df, {}, explore_layer_index))
    sc.stop()