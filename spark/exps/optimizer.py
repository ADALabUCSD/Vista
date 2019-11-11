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
sys.path.append('../code/python/cnn')
from vista import Vista

from pyspark.ml.classification import LogisticRegression, LinearSVC, DecisionTreeClassifier, GBTClassifier, RandomForestClassifier, MultilayerPerceptronClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
import time

# Example usage of the Vista optimizer for CNN feature transfer workload
if __name__ == '__main__':

    def downstream_ml_func(features_df, results_dict, layer_index, model_name='LogisticRegression', extra_config={}):

        def hyperparameter_tuned_model(clf, train_df):
            pipeline = Pipeline(stages=[clf])

            paramGrid = ParamGridBuilder()
            for i in extra_config:
                if i == 'numFolds':
                    continue
                paramGrid = paramGrid.addGrid(eval('clf.'+i), extra_config[i])

            paramGrid = paramGrid.build()

            if 'numFolds' in extra_config:
                numFolds = extra_config['numFolds']
            else:
                numFolds = 3 # default

            crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=numFolds)
            # Run cross-validation, and choose the best set of parameters.
            return crossval.fit(train_df)

        train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)

        if model_name == 'LogisticRegression':
            clf = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.1)

        if model_name == 'LinearSVC':
            clf = LinearSVC(maxIter=50, regParam=0.01)
        
        if model_name == 'DecisionTreeClassifier':
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(train_df)
            train_df = si_model.transform(train_df)
            
            clf = DecisionTreeClassifier(maxDepth=2, labelCol="indexed")

        if model_name == 'GBTClassifier':
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(train_df)
            train_df = si_model.transform(train_df)
            
            clf = GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, maxDepth=5)
        
        if model_name == 'RandomForestClassifier':
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(train_df)
            td = si_model.transform(train_df)
            
            clf = RandomForestClassifier(labelCol="label", featuresCol="features")
        
        if model_name == 'OneVsRest':
            lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.5)
            clf = OneVsRest(labelCol="label", featuresCol="features", predictionCol="prediction", classifier=lr)
        
        if extra_config != {}:
            model = hyperparameter_tuned_model(clf, train_df)
        else:
            model = clf.fit(train_df)

        predictions = model.transform(test_df)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                    metricName="accuracy")
        results_dict[layer_index] = evaluator.evaluate(predictions)
        return results_dict

    prev_time = time.time()
    # mem_sys_rsv is an optional parameter. If not set a default value of 3 will be used.
    vista = Vista("vista-example", 32, 8, 8, 'alexnet', 3, 0, 'hdfs://spark-master:9000/foods_sample.csv',
                      'hdfs://spark-master:9000/foods_images', 20129, 130, mem_sys_rsv=3, model_name='LogisticRegression', extra_config={})

    # Optional overrides
    vista.override_inference_type('bulk')
    vista.override_join('s')
    vista.overrdide_operator_placement('before-join')

    print(vista.run())
    print("Runtime: " + str((time.time()-prev_time)/60.0))
