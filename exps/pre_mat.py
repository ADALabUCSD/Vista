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
from pyspark.sql.functions import col

sys.path.append('../code/python')

from vista_utils import get_struct_df, get_images_df, get_dir_size
from vista_utils import image_to_byte_arr_udf
from vista_utils import get_image_features_for_layer

# Script for pre-materializing the CNN features of a base layer. CNN features will be stored
# in Parquet format on HDFS.
if __name__ == '__main__':
    ############################change appropriately###################################
    model = 'alexnet'
    pre_mat_layer_index = -4  # from the top
    images_input = 'hdfs://spark-cluster-master:9000/images'
    pre_mat_name = 'hdfs://spark-cluster-master:9000/' + model + "_pre_mat_layer.parquet"
    heap_memory = 29
    executor_cpu = 5
    sp_core_memory_fraction = 0.6
    ###################################################################################

    app_name = 'pre-mat-' + model + "-l:" + str(pre_mat_layer_index)
    conf = SparkConf()
    conf.setAppName(app_name)
    conf.set("spark.executor.memory", str(heap_memory) + "g")
    conf.set("spark.memory.fraction", sp_core_memory_fraction)
    conf.set("spark.executor.cores", executor_cpu)

    conf.set("spark.shuffle.reduceLocality.enabled", "false")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.files.maxPartitionBytes", "10485760")  # 10MB

    sc = SparkContext.getOrCreate(conf=conf)

    images_df = get_images_df(sc, images_input)
    images_df = images_df.select(col('id'), image_to_byte_arr_udf(sc, col('image_buffer')).alias('input_layer'))
    images_df = get_image_features_for_layer(model, pre_mat_layer_index, images_df, 0, False)[0].select("id", col(
        'image_features').alias('input_layer'))
    images_df.write.mode("overwrite").parquet(pre_mat_name)
    sc.stop()
