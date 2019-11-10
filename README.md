# Vista
Materialization Trade-offs for Feature Transfer from Deep CNNs for Multimodal Data Analytics [(technical report)](https://adalabucsd.github.io/papers/TR_2017_Vista.pdf). 

### Prerequisites
1. Spark cluster setuped with Spark standalone mode and HDFS (tested with Java 1.8).
2. TensorFlow installed (tested with TensorFlow 1.3.0)
3. Scala build tool (sbt) installed (tested with sbt version 0.13.9). For ubuntu-16.04 following code snippet will install sbt.
```bash
    echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
    sudo apt-get update
    sudo apt-get install sbt
```

### Project Structure
* /code: Contains python and scala code for Vista optimizer and helper functions.
* /data: Contains data pre-processing/generations scripts for the Foods and Amazon datasets.
* /exps: Contains example scripts on how to use the Vista optimizer and other scripts implementing baseline approaches
 discussed in the paper.
 
### How to use
1. Clone this repository in the Spark Master node.
```
    git clone https://github.com/ADALabUCSD/vista.git
```
2. Run the download_cnn_weights.sh script to get the pre-trained weights for the Convnets
```
    $ ./download_cnn_weights.sh
```
3. Go to the /data/{amazon or foods} directory and run the download_data.sh script and generate_data.py script.
```
    $ ./download_data.sh
    $ python generate_data.py
```
4. Ingest the generated strucutured data file (amazon.csv or foods.csv) and images into HDFS. Alternatively any other input data can be also used. Strucutured file should confirm to the {ID, X_str, y} format without the header and the images directory should contain the resized RGB images (227*227) named after the ID (e.g. ID.jpg).
```
    $ hadoop fs -put ./foods.csv    /foods.csv
    $ hadoop fs -put ./images       /images
```
5. Go to to /code/scala directory and build scala project to create a jar containing helper functions. The generated jar can be found at /code/scala/target/scala-2.11/vista-udfs_2.11-1.0.jar
```
    $ sbt package
```
6. Go to /exps directory and copy the optimizer.py to a different file. Change the content of the file for your requirement. The first important thing is creating an instance of Vista class by providing all the inputs and configuration values. After this the optimizer will make decisions and pick values for the logical plan, physical plan operators and Spark config values. Alternative the user can override the optimizer picked decisions.
```
    /** Instantiation Parameters
     * name         : Name given to the Spark job
     * mem_sys      : System memory of a Spark worker
     * n_nodes      : Number of nodes in the Spark cluster
     * cpu_sys      : Number of CPUs available in the Spark cluster
     * model        : ConvNet model name. Possible values -> {'alexnet', 'vgg16', 'resnet50'}
     * n_layers     : Number of layers from the top most layer of the ConvNet to be explored
     * start_layer  : Starting layer of the ConvNet. Use 0 when starting with raw images
     * ml_func      : Function pointer to the downstream ML model
     * struct_input : Input path to the strucutred input
     * images_input : Input path to the images
     * n            : Number of total records
     * dS           : number of structured features
     * model_name   : Name of the (PySpark MLLib) Downstream ML Model to run in the Vista optimizer
     * extra_config : Extra configuration settings for hyperparameter tuning with the downstream model
    **/
    vista = Vista("vista-example", 32, 8, 8, 'alexnet', 4, 0, downstream_ml_func, 'hdfs://../foods.csv',
                      'hdfs://.../images', 20129, 130, model_name='LogisticRegression', extra_config={})
    
    //possible values for model_name -> {'LogisticRegression', 'LinearSVC', 'DecisionTreeClassifier', 'GBTClassifier', 'RandomForestClassifier', 'OneVsRest'}
    
    // extra_config takes in a dictionary of chosen model's attribute names and list of values to explore as key-value pairs for k-Fold Cross Validation. It can also take `numFolds` for the k value. 
    // extra_config is applicable for all currently supported downstream models except 'OneVsRest'.

    //Optional: overriding system picked decisions
    vista.override_inference_type('bulk')               //posible value -> {'bulk', 'staged'}
    vista.overrdide_operator_placement('before-join')   //posible value -> {'before-join', 'after-join'}
    vista.override_join('s')                            //posible value -> {'b', 's'}
    vista.override_persistence_format('deser')          //posible value -> {'ser', 'deser'}
    
    //Starting the ConvNet feature transfer workload
    print(vista.run())
```
7. To submit the Spark job use the following command. We recommend using atleast 4GB of Spark driver memory. vista.py should be changed to point to the correct python script.
```
    $ spark-submit --master <spark-master-url> --driver-memory 8g --packages databricks:tensorframes:0.2.9-s_2.11 --jars ../code/scala/target/scala-2.11/vista-udfs_2.11-1.0.jar vista.py
```

### Limitations
* For the Conv layers when transferring features Vista applies max pooling by default. The filter widths and strides are selected such that every Conv volume will reduce into 2*2 filters with the same depth. Right now this configuration is not configurable. Ideally a user should be able specify different feature transformations on the Conv features such max/avg pooling.
