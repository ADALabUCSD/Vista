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

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Books.json.gz -P ./
gunzip -k meta_Books.json.gz
sed "s/'/\"/g" meta_Books.json > books_data.json
rm meta_Books.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Video_Games.json.gz -P ./
gunzip -k meta_Video_Games.json.gz
sed "s/'/\"/g" meta_Video_Games.json > video_data.json
rm meta_Video_Games.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Toys_and_Games.json.gz -P ./
gunzip -k meta_Toys_and_Games.json.gz
sed "s/'/\"/g" meta_Toys_and_Games.json > toys_data.json
rm meta_Toys_and_Games.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Tools_and_Home_Improvement.json.gz -P ./
gunzip -k meta_Tools_and_Home_Improvement.json.gz
sed "s/'/\"/g" meta_Tools_and_Home_Improvement.json > tools_data.json
rm meta_Tools_and_Home_Improvement.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Sports_and_Outdoors.json.gz -P ./
gunzip -k meta_Sports_and_Outdoors.json.gz
sed "s/'/\"/g" meta_Sports_and_Outdoors.json > sports_data.json
rm meta_Sports_and_Outdoors.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Kindle_Store.json.gz -P ./
gunzip -k meta_Kindle_Store.json.gz
sed "s/'/\"/g" meta_Kindle_Store.json > kindle_data.json
rm meta_Kindle_Store.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Health_and_Personal_Care.json.gz -P ./
gunzip -k meta_Health_and_Personal_Care.json.gz
sed "s/'/\"/g" meta_Health_and_Personal_Care.json > health_data.json
rm meta_Health_and_Personal_Care.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Cell_Phones_and_Accessories.json.gz -P ./
gunzip -k meta_Cell_Phones_and_Accessories.json.gz
sed "s/'/\"/g" meta_Cell_Phones_and_Accessories.json > cell_phone_data.json
rm meta_Cell_Phones_and_Accessories.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Electronics.json.gz -P ./
gunzip -k meta_Electronics.json.gz
sed "s/'/\"/g" meta_Electronics.json > home_data.json
rm meta_Electronics.json.gz*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz -P ./
gunzip -k meta_Clothing_Shoes_and_Jewelry.json.gz
sed "s/'/\"/g" meta_Clothing_Shoes_and_Jewelry.json > clothing_data.json
rm meta_Clothing_Shoes_and_Jewelry.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz -P ./
gunzip -k meta_Clothing_Shoes_and_Jewelry.json.gz
sed "s/'/\"/g" meta_Clothing_Shoes_and_Jewelry.json > electronics_data.json
rm meta_Clothing_Shoes_and_Jewelry.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_CDs_and_Vinyl.json.gz -P ./
gunzip -k meta_CDs_and_Vinyl.json.gz
sed "s/'/\"/g" meta_CDs_and_Vinyl.json > cds_data.json
rm meta_CDs_and_Vinyl.json*

wget http://jmcauley.ucsd.edu/data/amazon/categoryFiles/meta_Movies_and_TV.json.gz -P ./
gunzip -k meta_Movies_and_TV.json.gz
sed "s/'/\"/g" meta_Movies_and_TV.json.gz > movies_data.json
rm meta_Movies_and_TV.json.gz*