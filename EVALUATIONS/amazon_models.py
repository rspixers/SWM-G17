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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

!pip install pysentiment2

import pysentiment2 as ps 


amazon_labelled = pd.read_csv('../DATA/Amazon_processed.csv')

amazon_labelled = amazon_labelled[amazon_labelled.text.notnull()]

y = amazon_labelled['label'].values
# project_data.drop(['label'], axis=1, inplace=True)
print(amazon_labelled.head(5))
X = amazon_labelled.text.values

from sklearn.model_selection import train_test_split
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
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range = (2,2),max_features = 4000, min_df = 10)
X_train_preprocessed_text = X_train
X_test_preprocessed_text = X_test
X_train_2gram_features = vectorizer.fit_transform(X_train_preprocessed_text)

vectorizer_dict = {"vectorizer" : vectorizer}

X_test_2gram_features = vectorizer.transform(X_test_preprocessed_text)


import pickle
outfile = open('vectorizer_amazon','wb')
pickle.dump(vectorizer_dict,outfile)
outfile.close()


# 3 gram
vectorizer = CountVectorizer(ngram_range = (3,3),max_features = 4000, min_df = 10)
X_train_preprocessed_text = X_train
X_test_preprocessed_text = X_test
X_train_3gram_features = vectorizer.fit_transform(X_train_preprocessed_text)
X_test_3gram_features = vectorizer.transform(X_test_preprocessed_text)


import numpy as np
Sentiment_X_train_text = np.column_stack((Xtrain_positive, Xtrain_negative))
Sentiment_X_test_text = np.column_stack((Xtest_positive, Xtest_positive))


#Combining all features
X_train_2gram_features_array = X_train_2gram_features.toarray()
Xtrain = np.concatenate((Sentiment_X_train_text,X_train_2gram_features_array), axis = 1)
X_test_2gram_features_array = X_test_2gram_features.toarray()
Xtest = np.concatenate((Sentiment_X_test_text,X_test_2gram_features_array), axis = 1)


# SVM
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

def SVM_classifier():
    svclassifier = LinearSVC(loss='hinge', verbose=10, max_iter=100000)
    svclassifier.fit(Xtrain, y_train)
    y_pred = svclassifier.predict(Xtest)

    filename = 'LinearSVC_amazon.sav'
    pickle.dump(svclassifier, open(filename, 'wb'))
    
    
    print("Accuracy for SVM: "+ str(accuracy_score(y_test,y_pred)))
    print("Precision for SVM: "+ str(recall_score(y_test, y_pred, average='weighted')))
    print("Recall for SVM: "+ str(precision_score(y_test, y_pred, average='weighted')))
    print("F1 for SVM: "+ str(f1_score(y_test,y_pred)))


print(SVM_classifier())


from sklearn.linear_model import LogisticRegression
def logisticRegression():
    clf = LogisticRegression()
    clf.fit(Xtrain, y_train)
    y_pred = clf.predict(Xtest)
    filename = 'Logistic_Regression_Amazon.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
    print("Accuracy for Logistic Regression: "+ str(accuracy_score(y_test,y_pred)))
    print("Precision for Logistic Regression: "+ str(recall_score(y_test, y_pred, average='weighted')))
    print("Recall for Logistic Regression: "+ str(precision_score(y_test, y_pred, average='weighted')))
    print("F1 for Logistic Regression: "+ str(f1_score(y_test,y_pred)))

print(logisticRegression())


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
def naiveBayes():
    clf_NB = GaussianNB()
    clf_NB.fit(Xtrain,y_train)
    y_pred = clf_NB.predict(Xtest)
    filename = 'Naive_Bayes_Amazon.sav'
    pickle.dump(clf_NB, open(filename, 'wb'))
   
    print("Accuracy for Naive Bayes: "+ str(accuracy_score(y_test,y_pred)))
    print("Precision for Naive Bayes: "+ str(recall_score(y_test, y_pred, average='weighted')))
    print("Recall for Naive Bayes: "+ str(precision_score(y_test, y_pred, average='weighted')))
    print("F1 for Naive Bayes: "+ str(f1_score(y_test,y_pred)))


print(naiveBayes())

def mlp():
    clf = MLPClassifier()
    clf.fit(Xtrain, y_train)
    y_pred = clf.predict(Xtest)
    filename = 'MLP_Amazon.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Accuracy for MLP: "+ str(accuracy_score(y_test,y_pred)))
    print("Precision for MLP: "+ str(recall_score(y_test, y_pred, average='weighted')))
    print("Recall for MLP: "+ str(precision_score(y_test, y_pred, average='weighted')))
    print("F1 for MLP: "+ str(f1_score(y_test,y_pred)))

print(mlp())