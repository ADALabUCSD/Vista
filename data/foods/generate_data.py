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

#############################################################################################
#                                                                                           #
# This script generates the structured data file for Foods dataset.                         #
#                                                                                           #
#############################################################################################
import pandas as pd
from itertools import combinations
import numpy as np

df = pd.read_csv('foods_raw.tsv', sep='\t', header=0, index_col=None, dtype={'code': object}, low_memory=False)

df = df[df['image_url'].notnull()]
df = df.dropna(thresh=len(df)*.5, axis=1)
df = df.dropna()
df['code'] = df['code'].map(lambda x: x.lstrip('0+'))
df['veg'] = df['main_category_en'].map(lambda x: 1 if x == 'Plant-based foods and beverages' else 0)

df_new = df[['additives_n', 'ingredients_from_palm_oil_n', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
             'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'sodium_100g']]

#cc = list(combinations(df_new.columns, 2))
#temp1 = pd.concat([df_new[c[1]].mul(df_new[c[0]]) for c in cc], axis=1, keys=cc)

#cc = list(combinations(df_new.columns, 3))
#temp2 = pd.concat([df_new[c[2]] * df_new[c[1]] * df_new[c[0]] for c in cc], axis=1, keys=cc)

#df_new = pd.concat([df[['code']], df_new, temp1, temp2, df[['veg']]], axis=1)
#df_new.to_csv('./foods.csv', header=None, index=False)

df_new = pd.concat([df[['code']], df_new, df[['veg']]], axis=1)
df_new.to_csv('./foods.csv', header=None, index=False)

# train dataset
msk1 = np.random.rand(len(df)) < 0.6
df_new[msk1].to_csv('./train_foods.csv', header=None, index=False)

# validation dataset
msk2 = np.random.rand(len(df_new[~msk1])) < 0.5
df_new[~msk1][msk2].to_csv('./validation_foods.csv', header=None, index=False)
# test dataset
df_new[~msk1][~msk2].to_csv('./test_foods.csv', header=None, index=False)
