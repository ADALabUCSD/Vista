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

import time
import sys
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, broadcast, lit, array

sys.path.append('../code/python')

from vista_utils import get_struct_df, get_images_df, get_dir_size
from vista_utils import image_to_byte_arr_udf, get_joined_features, downstream_ml_func
from vista_utils import get_image_features_for_layer, get_feature_projections

# Baseline approach for CNN feature transfer workloads. CNN features for each interested
# layer is first materialized and downstream ML model is evaluated for each layer separately.
if __name__ == '__main__':

    ############################change appropriately###################################
    model = 'resnet50'
    layer_index = -4  # from the top
    # images_input = 'hdfs://spark-cluster-master:9000/images'
    # struct_input = 'hdfs://spark-cluster-master:9000/foods.csv'
    images_input = 'file:///home/snakanda/Work/Vista/data/foods/images'
    struct_input = 'file:///home/snakanda/Work/Vista/data/foods/foods.csv'
    heap_memory = 25
    num_executors = 1
    executor_cpu = 1
    storage_level = StorageLevel(True, True, False, True)# memory and disk deserialized
    sp_core_memory_fraction = 0.6
    ###################################################################################

    prev_time = time.time()
    app_name = 'baseline-' + model + "-l:" + str(layer_index)
    conf = SparkConf()
    conf.setAppName(app_name)
    conf.set("spark.executor.memory", str(heap_memory) + "g")
    conf.set("spark.memory.fraction", sp_core_memory_fraction)
    conf.set("spark.executor.cores", executor_cpu)
    conf.set("spark.cores.max", num_executors*executor_cpu)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.reduceLocality.enabled", "false")
    conf.set("spark.files.maxPartitionBytes", "10485760")  # 10MB

    sc = SparkContext.getOrCreate(conf=conf)
    sql_context = SQLContext(sc)

    struct_df = get_struct_df(sc, struct_input)
    images_df = get_images_df(sc, images_input)
    images_df = images_df.select(col('id'), image_to_byte_arr_udf(sc, col('image_buffer')).alias('input_layer'))

    image_features_df, shape = get_image_features_for_layer(model, layer_index, images_df, 0, False)
    features_df = get_joined_features(image_features_df, struct_df, False)
    features_df = features_df.select("id", "features", "image_features", "label")
    merged_features_df = get_feature_projections(sc, features_df, 1, [shape])[0]

    merged_features_df._jdf.persist(sc._getJavaStorageLevel(storage_level))

    print(downstream_ml_func(merged_features_df, {}, layer_index))
    sc.stop()
    print("Runtime: " + str((time.time()-prev_time)/60.0))
