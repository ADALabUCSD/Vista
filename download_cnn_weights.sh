#Copyright 2018 Supun Nakandala and Arun Kumar
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

#!/usr/bin/env bash
#downloading ConvNet pre-trained weights
mkdir -p ./code/python/cnn/resources
wget http://cseweb.ucsd.edu/~snakanda/downloads/vista/cnn-weights/alexnet_weights.h5 -P ./spark/code/python/cnn/resources/
wget http://cseweb.ucsd.edu/~snakanda/downloads/vista/cnn-weights/vgg16_weights.h5 -P ./spark/code/python/cnn/resources/
wget http://cseweb.ucsd.edu/~snakanda/downloads/vista/cnn-weights/resnet50_weights.h5 -P ./spark/code/python/cnn/resources/