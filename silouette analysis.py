
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import csv
import numpy as np
import string
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from time import time
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import pandas as pd
from io import open


fname='/Users/yisili/Documents/IBM/projects/schlumberger oil company/data validation analysis/data.xlsx'

data_file = pd.ExcelFile(fname) ### reading the excel file as an xlsx object for fast reading in future

train_df = data_file.parse('train', names = None)

train_df=train_df.fillna('')
# print df.ix[:5, :4]
# df = df.ix[:5, :4]
train_data = train_df.values.tolist()

table = string.maketrans("","")
documents=[]
for i in range(1,len(train_data)):
### you are starting w 1 b/c the i=0 is where you have header (column name row);

    documents = documents + (str(train_data[i]).translate(table, string.punctuation).split(',', len(train_data) - 1))

true_k = range(2,10,1)
# true_k = 4

for k in true_k:
    vectorizer = TfidfVectorizer(vocabulary=None)
    X = vectorizer.fit_transform(documents)

    ### now use pretty much any models of your choice to analyze the data
    # we'll first try kmeans (then move on something else?
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=1010)

    model.fit(X)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()  ### basically get the terms in the documents

    no_features = len(terms)   ### total # of terms: 143
    # print no_features

    # For info, print out the top 4
    top = 4
    sample_size = len(train_df)

    # First score for the unreduced data - WILL NOT BE USED IN DEMO

    print(79 * '_')
    print("Silhouette Coefficient: %0.3f"
         % metrics.silhouette_score(X, model.labels_,sample_size=sample_size))



