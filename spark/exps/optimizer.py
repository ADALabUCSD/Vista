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

import sys
sys.path.append('../code/python')
from vista import Vista

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

# Example usage of the Vista optimizer for CNN feature transfer workload
if __name__ == '__main__':

    def downstream_ml_func(features_df, results_dict, layer_index):
        lr = LogisticRegression(labelCol="label", featuresCol="features", elasticNetParam=0.5, regParam=0.01)
        train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)
        #valid_df, test_df = test_df.randomSplit([0.5, 0.5], seed=2018)

        model = lr.fit(train_df)
        result = {}

        predictions = model.transform(train_df)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="f1")
        result['train_acc'] = evaluator.evaluate(predictions)

        #predictions = model.transform(valid_df)
        #evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
        #                                             metricName="f1")
        #result['valid_acc'] = evaluator.evaluate(predictions)

        predictions = model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="f1")
        result['test_acc'] = evaluator.evaluate(predictions)

        results_dict[layer_index] = result
        return results_dict
    
#     vista = Vista("vista-example", 16, 8, 8, 'alexnet', 4, 0, downstream_ml_func, 'hdfs://spark-cluster-master:9000/amazon.csv', 'hdfs://spark-cluster-master:9000/amazon_images', 20000, 100, mem_sys_rsv=3)

#     result = vista.run()
#     for k in [-1, -2, -3, -4]:
#         print(k, result[k])

    
    import pyspark
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.types import StringType, FloatType, IntegerType
    
    toDense = F.udf(lambda v: Vectors.dense(v.toArray()), VectorUDT())
    
    sc = pyspark.SparkContext(appName="vista")
    
    
    def get_struct_df(sc, data_file_path):
        sql_context = pyspark.SQLContext(sc)
        struct_df = sql_context.read.format('csv').options(header='false', inferSchema='true').load(data_file_path)
        col_names = struct_df.schema.names

        struct_df = VectorAssembler(inputCols=col_names[1:-1], outputCol="features").transform(struct_df)
        struct_df = struct_df.withColumn("features", toDense("features"))
        struct_df = struct_df.withColumn("id", struct_df[col_names[0]].cast(StringType())) \
            .withColumn("label", struct_df[col_names[-1]].cast(IntegerType())) \
            .select("id", "features", "label")
        return struct_df
    
    
    
#     features_df = get_struct_df(sc, 'hdfs://spark-cluster-master:9000/amazon.csv')
#     print(downstream_ml_func(features_df, {}, 0))

    def get_hog_df(sc, data_file_path):
        sql_context = pyspark.SQLContext(sc)
        struct_df = sql_context.read.format('csv').options(header='false', inferSchema='true').load(data_file_path)

        col_names = struct_df.schema.names

        struct_df = VectorAssembler(inputCols=col_names[:-1], outputCol="features").transform(struct_df)
        struct_df = struct_df.withColumn("features", toDense("features"))
        struct_df = struct_df.withColumn("label", struct_df[col_names[-1]].cast(IntegerType())) \
            .select("features", "label")
        return struct_df
    
    hog_df = get_hog_df(sc, 'hdfs://spark-cluster-master:9000/amazon_with_hog.csv')
    print(downstream_ml_func(hog_df, {}, 0))