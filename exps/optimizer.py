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


# Example usage of the Vista optimizer for CNN feature transfer workload
if __name__ == '__main__':

    def downstream_ml_func(features_df, results_dict, layer_index):
        lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.5)
        model = lr.fit(features_df)
        predictions = model.transform(features_df)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        results_dict[layer_index] = evaluator.evaluate(predictions)
        return results_dict

    # mem_sys_rsv is an optional parameter. If not set a default value of 3 will be used.
    vista = Vista("vista-example", 32, 8, 8, 'alexnet', 4, 0, downstream_ml_func, 'hdfs://spark-cluster-master:9000/foods.csv',
                      'hdfs://spark-cluster-master:9000/images', 20129, 130, mem_sys_rsv=3)

    # Optional overrides
    vista.override_inference_type('bulk')
    vista.override_join('s')
    vista.overrdide_operator_placement('before-join')

    print(vista.run())