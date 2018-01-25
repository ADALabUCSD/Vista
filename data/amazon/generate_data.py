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
# This script generates the structured data file for Amazon dataset and crawls the images.  #
#                                                                                           #
#############################################################################################
import json
import urllib
from PIL import Image
import pandas as pd

import gensim
import nltk
import numpy as np
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA

raw_data_files = ['clothing_data.json', 'toys_data.json', 'sports_data.json', 'video_data.json', 'tools_data.json',
                  'kindle_data.json', 'health_data.json', 'cell_phone_data.json', 'home_data.json',
                  'books_data.json', 'electronics_data.json', 'cds_data.json', 'movies_data.json']

max_num_of_records = 200000

nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])


# This function does all cleaning of data using two objects above
def nlp_clean(data):
    new_data = []
    for d in data:
        new_str = d.lower()
        dlist = tokenizer.tokenize(new_str)
        dlist = list(set(dlist).difference(stopword_set))
        new_data.append(dlist)
    return new_data


if __name__ == '__main__':
    line_count = 0
    invalid_line_count = 0
    brands = {}
    categories = {}
    item_dicts = []

    rank_sum = 0.0

    for file in raw_data_files:
        with open(file) as json_file:
            lines = json_file.readlines()

            for line in lines:
                try:
                    data = json.loads(line)
                    rank = list(data['salesRank'].values())[0]
                    title = data['title']
                    price = data['price']

                    temp_cats = []
                    for cat in data['categories'][0]:
                        if cat not in categories:
                            categories[cat] = len(categories)
                        temp_cats.append(categories[cat])

                    item = {
                        'id': data['asin'],
                        'price': price,
                        'rank': rank,
                        'title': title,
                        'categories': temp_cats
                    }
                    url = data['imUrl']
                    extension = url.split(".")[-1]
                    if extension == "jpg":  # and not os.path.isfile("./images/" + data['asin'] + "." + extension):
                        img_file = urllib.URLopener()
                        img_file.retrieve(url, "temp." + extension)
                        image = Image.open("temp." + extension).resize((227, 227))
                        image.save("./images/" + data['asin'] + "." + extension)

                        item_dicts.append(item)
                        line_count += 1
                        rank_sum = rank_sum + int(rank)

                    # if line_count % 10 == 0:
                    #    print('line count: ' + str(line_count))
                    if line_count >= max_num_of_records:
                        break

                except:
                    invalid_line_count += 1
                    # print "json_parsing_failed"
                    # tb = traceback.format_exc()
                    # print(tb)
                    # if invalid_line_count % 1000 == 0:
                    # print('invalid line count: ' + str(invalid_line_count))

            size = len(item_dicts)
            # print(size)
            # print(rank_sum/size)
            # print('size of the data: ' + str(size) + " records")
            # print('invalid records count:' + str(invalid_line_count))

            text_data = [x['title'] for x in item_dicts]
            text_data = nlp_clean(text_data)

    it = LabeledLineSentence(text_data, [str(x) for x in range(size)])
    model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    model.train(it, total_examples=model.corpus_count, epochs=10)

    doc_vecs = [model.docvecs[x] for x in range(size)]

    cats_size = len(categories)
    cats_vecs = []
    for x in range(size):
        temp = [0] * cats_size
        for y in item_dicts[x]['categories']:
            temp[y] = 1
        cats_vecs.append(temp)
    cats_vecs = np.array(cats_vecs, dtype=np.float64)
    if cats_size > 100:
        print('dim. reducing category vectors')
        cats_vecs = PCA(n_components=100).fit_transform(cats_vecs)

    with open('amazon.csv', 'w') as file_out:
        for item, doc_vec, cats_vec in zip(item_dicts, doc_vecs, cats_vecs):
            file_out.write(item['id'] + ","
                           + str(item['price']) + ","
                           + ",".join([str(x) for x in doc_vec.tolist()]) + ","
                           + ",".join([str(x) for x in cats_vec.tolist()]) + ",")
            if int(item['rank']) < rank_sum / size:
                file_out.write("0\n")
            else:
                file_out.write("1\n")

df_new = pd.read_csv('amazon.csv', header=None)

# train dataset
msk1 = np.random.rand(len(df_new)) < 0.6
df_new[msk1].to_csv('./train_amazon.csv', header=None, index=False)

# validation dataset
msk2 = np.random.rand(len(df_new[~msk1])) < 0.5
df_new[~msk1][msk2].to_csv('./validation_amazon.csv', header=None, index=False)
# test dataset
df_new[~msk1][~msk2].to_csv('./test_amazon.csv', header=None, index=False)