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
import spacy

import en_core_web_lg
nlp = en_core_web_lg.load()
global count
count = 0


def filterStopwordsAndLemmatize(text):
    global count
    count += 1
    if count % 1000 == 0:
        print(count)
    doc = nlp(text)
    temp = ""
    for token in doc:
        if not token.is_stop:
            temp += token.lemma_ + " "
    return temp


#amazon5 = pd.read_csv("data/amazon5.csv")
amazon5['lemmatized_text'] = amazon5['filteredtext'].apply(
    filterStopwordsAndLemmatize)
amazon5.to_csv("data/amazon5_lemmatized.csv")
