import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from nltk.stem import PorterStemmer
import pandas as pd

# more efficient library (less accurate)
nltk.download("words")
# more robust library (slower)
nltk.download("brown")

import pandas as pd

global count
count = 0


def stemming(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

# calculate the percentage of stemmed words found in the english dictionary
def english_filter(data):
    preparse_length = len(word_tokenize(str(data)))
    parsed = ""
    global count
    english_dictionary = set(nltk.corpus.words.words())
    if preparse_length:
        parsed = " ".join(
            w for w in word_tokenize(str(data)) if w in english_dictionary
        )
        parsed_length = len(parsed.split())
        if parsed_length / preparse_length < 0.2:
            count = count + 1
            parsed = ""
    print(parsed_length / preparse_length)
    return parsed


filtered = pd.DataFrame()
unfiltered = pd.DataFrame()
amazon5_df = pd.read_csv("data/amazon5.csv")
print(amazon5_df.head())
words = set(nltk.corpus.words.words())

for index, text in enumerate(amazon5_df["filteredtext"]):
    sent = text
    sent_stemmed = stemming(sent)
    # print(sent_stemmed)
    parsed = english_filter(sent_stemmed)
    if parsed == "":
        # print(amazon5_df.iloc[index])
        filtered = filtered.append(amazon5_df.iloc[index])
    # print(parsed)
    else:
        unfiltered = unfiltered.append(amazon5_df.iloc[index])
print(filtered)
filtered.to_csv("data/filtered.csv", encoding="utf-8")
unfiltered.to_csv("data/unfiltered.csv", encoding="utf-8")
print(count)
print(len(amazon5_df))
