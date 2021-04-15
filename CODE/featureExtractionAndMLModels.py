#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:07:08 2021

@author: mageshsridhar
"""

import pandas as pd
import numpy as np
import os
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

def tfIDFvectorization(input_df):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(list(input_df["filteredtext"]))
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return df

def logisticRegression():
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))
    print("LR Training Score: ",clf.score(X_train,y_train))
    print("LR F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    print("LR Accuracy:",accuracy_score(y_test,clf.predict(X_test)))

def naiveBayes():
    clf_NB = GaussianNB()
    clf_NB.fit(X_train,y_train)
    print(classification_report(y_test, clf_NB.predict(X_test)))
    print("NB Training Score:",clf_NB.score(X_train,y_train))
    print("NB F1 Score: ",f1_score(y_test,clf_NB.predict(X_test),zero_division=0))
    print("NB Accuracy:",accuracy_score(y_test,clf_NB.predict(X_test)))

def mlp():
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))
    print("MLP Training Score: ",clf.score(X_train,y_train))
    print("MLP F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    print("MLP Accuracy:",accuracy_score(y_test,clf.predict(X_test)))

def linearSVM():
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))
    print("Linear SVM Training Score: ",clf.score(X_train,y_train))
    print("Linear SVM F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    print("Linear Accuracy:",accuracy_score(y_test,clf.predict(X_test)))

def randomForest():
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))
    print("RF Training Score: ",clf.score(X_train,y_train))
    print("RF F1 Score:",f1_score(y_test,clf.predict(X_test),zero_division=0))
    print("RF Accuracy:",accuracy_score(y_test,clf.predict(X_test)))

amazon5 = pd.read_csv("data/amazon5.csv")
tfidf_df = tfIDFvectorization(amazon5)
print(tfidf_df)
tfidf_df = normalize(tfidf_df)
pca = PCA(n_components=3000)
amazon5_tfidf_reduced = pca.fit_transform(tfidf_df)
labels = amazon5["label"]
plt.scatter(pd.DataFrame(amazon5_tfidf_reduced)[0],pd.DataFrame(amazon5_tfidf_reduced)[1],c=labels)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(amazon5_tfidf_reduced, labels, test_size=0.2, random_state=42)
logisticRegression()
print()
naiveBayes()
print()
mlp()
print()
linearSVM()
print()
randomForest()
