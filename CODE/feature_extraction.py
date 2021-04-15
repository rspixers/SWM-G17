from typing import Counter
import pandas as pd
import numpy as np
import time
import collections
from nltk import word_tokenize
from nltk.util import ngrams


start = time.time()


def calc_ngram(text, n):
    token = word_tokenize(text)
    return list(ngrams(token, n))


data = pd.read_csv("data/amazon5.csv", index_col=False)
_2pos_grams_df = data[data["label"] == 1]
_2pos_grams = _2pos_grams_df["filteredtext"].apply(calc_ngram, n=2)
_2pos_grams = [item for sublist in _2pos_grams for item in sublist]

_2neg_grams_df = data[data["label"] == -1]
_2neg_grams = _2neg_grams_df["filteredtext"].apply(calc_ngram, n=2)
_2neg_grams = [item for sublist in _2neg_grams for item in sublist]

_2pos_gram_counts = Counter(_2pos_grams)
print(_2pos_gram_counts.most_common(5))
_2neg_gram_counts = Counter(_2neg_grams)
print(_2neg_gram_counts.most_common(5))

_2ovr_gram_counts = Counter()
_2ovr_gram_counts.update(_2pos_gram_counts)
_2ovr_gram_counts.subtract(_2neg_gram_counts)

print(_2ovr_gram_counts)

end = time.time()
print(end - start)
