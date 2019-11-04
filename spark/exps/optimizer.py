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

    def downstream_ml_func(features_df, results_dict, layer_index, model_name='LogisticRegression'):
        #----------------------- Logistic Regression ------------------------------
	if model_name == 'LogisticRegression':
	    train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)

            lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.1)
            pipeline = Pipeline(stages=[lr])

            paramGrid = ParamGridBuilder() \
                .addGrid(lr.maxIter, [10, 20, 50]) \
                .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
                .build()

            crossval = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=MulticlassClassificationEvaluator(),
                                      numFolds=3)

            # Run cross-validation, and choose the best set of parameters.
            model = crossval.fit(train_df)

            predictions = model.transform(test_df)
        # ------------------------------------------------------------------------
        
        #------------------------------- Linear SVC -----------------------------------
        if model_name == 'LinearSVC':
	    svm = LinearSVC(maxIter=5, regParam=0.01)
            train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)
            pipeline = Pipeline(stages=[svm])

            paramGrid = ParamGridBuilder() \
                          .addGrid(svm.regParam, [0.01, 0.1, 0.5]) \
                          .addGrid(svm.maxIter, [5, 10, 20]) \
                          .build()

            crossval = CrossValidator(estimator=pipeline,\
                          estimatorParamMaps=paramGrid,\
                          evaluator=MulticlassClassificationEvaluator(),\
                          numFolds=3)

            model = crossval.fit(train_df)

            predictions = model.transform(test_df)
        # --------------------------------------------------------------------------
        
        #------------------------------- Decision Tree -----------------------------------
        if model_name == 'DecisionTreeClassifier':
	    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)
            si_model = stringIndexer.fit(train_df)
            td = si_model.transform(train_df)
            dt = DecisionTreeClassifier(maxDepth=2, labelCol="indexed")

            pipeline = Pipeline(stages=[dt])

            paramGrid = ParamGridBuilder() \
                          .addGrid(dt.maxDepth, [2, 5, 10, 15, 20, 25, 30]) \
                          .build()

            crossval = CrossValidator(estimator=pipeline,\
                          estimatorParamMaps=paramGrid,\
                          evaluator=MulticlassClassificationEvaluator(),\
                          numFolds=3)

            model = crossval.fit(td)

            #model = dt.fit(td)
            predictions = model.transform(test_df)
        # ---------------------------------------------------------------------------------

	#----------------------- GBT Classifier  ------------------------------
        if model_name == 'GBTClassifier':
	    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)
            si_model = stringIndexer.fit(train_df)
            td = si_model.transform(train_df)
            gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, maxDepth=5)

            pipeline = Pipeline(stages=[gbt])

            paramGrid = ParamGridBuilder() \
                          .addGrid(gbt.maxDepth, [2, 5, 10, 20, 30]) \
                          .addGrid(gbt.maxIter, [5, 10, 20, 50]) \
                          .build()

            crossval = CrossValidator(estimator=pipeline,\
                          estimatorParamMaps=paramGrid,\
                          evaluator=MulticlassClassificationEvaluator(),\
                          numFolds=3)

            model = crossval.fit(td)

            predictions = model.transform(test_df)
        # ------------------------------------------------------------------------        
       
	#----------------------- Random Forest Classifier  ------------------------------
        if model_name == 'RandomForestClassifier':
	    stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)
            si_model = stringIndexer.fit(train_df)
            td = si_model.transform(train_df)
            rfc = RandomForestClassifier(labelCol="label", featuresCol="features")

            pipeline = Pipeline(stages=[rfc])

            paramGrid = ParamGridBuilder() \
                          .addGrid(rfc.maxDepth, [5, 10, 20, 30]) \
                          .addGrid(rfc.numTrees, [10, 20, 50]) \
                          .build()

            crossval = CrossValidator(estimator=pipeline,\
                          estimatorParamMaps=paramGrid,\
                          evaluator=MulticlassClassificationEvaluator(),\
                          numFolds=3)

            model = crossval.fit(td)

            predictions = model.transform(test_df)
        # ------------------------------------------------------------------------ 
	
	#----------------------- OneVsRest  ------------------------------
        if model_name == 'OneVsRest':
	    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.5)
            train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=2019)
	    ovr = OneVsRest(labelCol="label", featuresCol="features", predictionCol="prediction", classifier=lr)
            model = ovr.fit(train_df)
            predictions = model.transform(test_df)
        # ------------------------------------------------------------------------
	 
	    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        results_dict[layer_index] = evaluator.evaluate(predictions)
        return results_dict

    prev_time = time.time()
    # mem_sys_rsv is an optional parameter. If not set a default value of 3 will be used.
    vista = Vista("vista-example", 32, 8, 8, 'alexnet', 3, 0, downstream_ml_func, 'hdfs://spark-master:9000/foods_sample.csv',
                      'hdfs://spark-master:9000/foods_images', 20129, 130, mem_sys_rsv=3, model_name='LogisticRegression')

    # Optional overrides (may uncomment/choose override option of any/all as needed)
    #vista.override_inference_type('bulk')
    #vista.override_join('s')
    #vista.overrdide_operator_placement('before-join')

    print(vista.run())
    print("Runtime: " + str((time.time()-prev_time)/60.0))
