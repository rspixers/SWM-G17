import pandas as pd
import numpy as np
import os

from IPython.display import display
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import scipy as sp
import pickle
from sklearn.model_selection import train_test_split



#!pip install pysentiment2

import pysentiment2 as ps 



amazon_data = pd.read_csv('./data/Amazon_processed.csv')
apple_data = pd.read_csv('./data/Apple_Processed.csv')


amazon_data = amazon_data[amazon_data.text.notnull()]
apple_data = apple_data[apple_data.text.notnull()]
resultant_data = pd.concat([amazon_data, apple_data])

y = resultant_data['label'].values
X = resultant_data.text.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)



# polarity
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def calculate_polarity(input_val):
    positive = []
    negative = []


    senti = SentimentIntensityAnalyzer()
    for sentence in input_val:
        kvp = senti.polarity_scores(sentence)
        positive.append(kvp['pos'])
        negative.append(kvp['neg'])


    return positive, negative



Xtrain_positive,Xtrain_negative = calculate_polarity(X_train)

Xtest_positive,Xtest_positive = calculate_polarity(X_test)



# 2 gram
vectorizer = CountVectorizer(ngram_range = (2,2),max_features = 4000, min_df = 10)
X_train_processed_text = X_train
X_test_processed_text = X_test
X_train_2gram = vectorizer.fit_transform(X_train_processed_text)

vectorizer_dict = {"vectorizer" : vectorizer}

X_test_2gram = vectorizer.transform(X_test_processed_text)


# 3 gram
vectorizer = CountVectorizer(ngram_range = (3,3),max_features = 4000, min_df = 10)
X_train_processed_text = X_train
X_test_processed_text = X_test
X_train_3gram_features = vectorizer.fit_transform(X_train_processed_text)
X_test_3gram_features = vectorizer.transform(X_test_processed_text)


Sentiment_X_train_text = np.column_stack((Xtrain_positive, Xtrain_negative))
Sentiment_X_test_text = np.column_stack((Xtest_positive, Xtest_positive))


#Features aggregation
X_train_2gram_array = X_train_2gram.toarray()
Xtrain = np.concatenate((Sentiment_X_train_text,X_train_2gram_array), axis = 1)
X_test_2gram_array = X_test_2gram.toarray()
Xtest = np.concatenate((Sentiment_X_test_text,X_test_2gram_array), axis = 1)

def LinearSVC():
    svc = LinearSVC(loss='hinge', verbose=10, max_iter=100000)
    svc.fit(Xtrain, y_train)
    y_pred = svc.predict(Xtest)
    return accuracy_score(y_test,y_pred)



def randomForest():
    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(Xtrain, y_train)
    y_pred=clf.predict(Xtest)
    # print(classification_report(y_test, clf.predict(X_test)))
    # print("RF Training Score: ",clf.score(X_train,y_train))
    # print("RF F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    # print("RF Accuracy:",accuracy_score(y_test,clf.predict(X_test)))
    return accuracy_score(y_test,y_pred)

print(randomForest())


from sklearn.linear_model import LogisticRegression
def logisticRegression():
    clf = LogisticRegression()
    clf.fit(Xtrain, y_train)
    # print(classification_report(y_test, clf.predict(X_test)))
    # print("LR Training Score: ",clf.score(X_train,y_train))
    # print("LR F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    # print("LR Accuracy:",accuracy_score(y_test,clf.predict(X_test)))
    return accuracy_score(y_test,clf.predict(Xtest))

print(logisticRegression())

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
def naiveBayes():
    clf_NB = GaussianNB()
    clf_NB.fit(Xtrain,y_train)
    # print(classification_report(y_test, clf_NB.predict(X_test)))
    # print("NB Training Score:",clf_NB.score(X_train,y_train))
    # print("NB F1 Score: ",f1_score(y_test,clf_NB.predict(X_test),zero_division=0))
    # print("NB Accuracy:",accuracy_score(y_test,clf_NB.predict(X_test)))
    return accuracy_score(y_test,clf_NB.predict(Xtest))


print(naiveBayes())


def mlp():
    clf = MLPClassifier()
    clf.fit(Xtrain, y_train)
    # print(classification_report(y_test, clf.predict(X_test)))
    # print("MLP Training Score: ",clf.score(X_train,y_train))
    # print("MLP F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    # print("MLP Accuracy:",accuracy_score(y_test,clf.predict(X_test)))
    return accuracy_score(y_test,clf.predict(Xtest))


print(mlp())

