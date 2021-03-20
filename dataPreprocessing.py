import os
import glob
import json
import pytz
import nltk

nltk.download("words")
nltk.download("brown")
nltk.download("punkt")
import numpy as np
import pandas as pd
import dateutil.parser

# from textblob import TextBlob
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize

import re
import nltk.data

from nltk.stem import PorterStemmer

import time
import string

import en_core_web_lg


start = time.time()

global words
words = set(nltk.corpus.brown.words())
words.add("aapl")
words.add("amzn")
words.add("amazon")
times = []
text = []
news_file_loc = []
sites = []
organizations = []
titles = []
published_times = []
time_zone = pytz.timezone("GMT")
news_path = "./data/News/"
json_files = glob.glob(news_path + "*/*.json")
print(len(json_files))

for news in json_files:
    with open(news, "r", encoding="utf-8") as f:
        record = json.load(f)
        local_time, offset = record["published"].split("+")
        local_datetime = datetime.strptime(local_time, "%Y-%m-%dT%H:%M:%S.%f")
        offset_datetime = datetime.strptime(offset, "%H:%M")
        published_times.append(
            local_datetime
            - timedelta(hours=offset_datetime.hour, minutes=offset_datetime.minute)
        )
        iso_time = dateutil.parser.isoparse(record["published"]).astimezone(time_zone)
        times.append(iso_time)
        news_file_loc.append(news)
        text.append(record["text"])
        sites.append(record["thread"]["site"])
        titles.append(record["title"])

news_df = pd.DataFrame(
    {
        "timestamp": times,
        "published": published_times,
        "text": text,
        "site": sites,
        "news_file_loc": news_file_loc,
    },
)


tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

nlp = en_core_web_lg.load()
stopwords = nlp.Defaults.stop_words


def filter_only_mention(text, stock, company):
    filterlist = []
    sentences = tokenizer.tokenize(text)
    if text.startswith("Official Notice Nr"):
        return None
    for sent in sentences:
        sent = sent.lower()
        sent = sent.replace("\n", " ")
        sent = re.sub(r"[^A-Za-z \. \:]+", "", sent)
        #         sent = re.sub(' +', ' ', sent)
        #         sent = re.sub('\.+', '.', sent)
        if (stock in sent) or (company in sent):
            filterlist.append(sent)
    if len(filterlist) == 0:
        return None
    filtered_text = " ".join(filterlist)
    return filtered_text


def filterStopwords(text):
    return " ".join([w for w in text.split() if w not in stopwords])


def lemmatize(text):
    return " ".join([lemma.lemma_ if lemma.lemma_ else text for lemma in nlp(text)])


def filter_only_language(mention_filtered_text):
    mention_filtered_text = mention_filtered_text.translate(
        str.maketrans("", "", string.punctuation)
    )
    # mention_filtered_text = mention_filtered_text.translate(
    #     str.maketrans("", "", string.digits)
    # )
    mention_filtered_list = word_tokenize(str(mention_filtered_text))
    global words
    preparse_length = float(len(mention_filtered_list))
    if preparse_length:
        parsed_filter_list = [w for w in mention_filtered_list if w in words]
        parsed_length = float(len(parsed_filter_list))
        # 80%, min=5 -> ~1300 records for amazon5_df : 400s
        # 50%, min=5 -> ~4800 records ...            : 900s
        # 50%, min=10 -> ~4300 records ...           : 900s
        # 25%, min=10 -> ~4500 records ...           : 950s
        if parsed_length / preparse_length < 0.5 or parsed_length < 10:
            mention_filtered_text = None
        else:
            mention_filtered_text = " ".join(parsed_filter_list)
    else:
        mention_filtered_text = None
    return mention_filtered_text


print(len(news_df))
news_df.drop_duplicates(subset=["timestamp", "published", "text", "site"], inplace=True)
print(len(news_df))
news_df.to_csv("data/news_df.csv", encoding="utf-8", index=False)

amazon_df = news_df.copy(deep=True)
apple_df = news_df.copy(deep=True)

amazon_df["filteredtext"] = amazon_df["text"].apply(
    filter_only_mention, args=("amzn", "amazon")
)
amazon_df.dropna(subset=["filteredtext"], inplace=True)
amazon_df["filteredtext"] = amazon_df["filteredtext"].apply(filterStopwords)
amazon_df.dropna(subset=["filteredtext"], inplace=True)
amazon_df["filteredtext"] = amazon_df["filteredtext"].apply(filter_only_language)
amazon_df.dropna(subset=["filteredtext"], inplace=True)
amazon_df["filteredtext"] = amazon_df["filteredtext"].apply(lemmatize)
print("before last drop", len(amazon_df))
amazon_df.dropna(subset=["filteredtext"], inplace=True)
print("after last drop", len(amazon_df))


apple_df["filteredtext"] = apple_df["text"].apply(
    filter_only_mention, args=("aapl", "apple")
)
apple_df.dropna(subset=["filteredtext"], inplace=True)
apple_df["filteredtext"] = apple_df["filteredtext"].apply(filterStopwords)
apple_df.dropna(subset=["filteredtext"], inplace=True)
apple_df["filteredtext"] = apple_df["filteredtext"].apply(filter_only_language)
apple_df.dropna(subset=["filteredtext"], inplace=True)
apple_df["filteredtext"] = apple_df["filteredtext"].apply(lemmatize)
apple_df.dropna(subset=["filteredtext"], inplace=True)

# Just for safety :P
amazon5_df = amazon_df.copy(deep=True)
amazon15_df = amazon_df.copy(deep=True)
amazon30_df = amazon_df.copy(deep=True)
amazon60_df = amazon_df.copy(deep=True)
amazon240_df = amazon_df.copy(deep=True)
amazon1440_df = amazon_df.copy(deep=True)


apple5_df = apple_df.copy(deep=True)
apple15_df = apple_df.copy(deep=True)
apple30_df = apple_df.copy(deep=True)
apple60_df = apple_df.copy(deep=True)
apple240_df = apple_df.copy(deep=True)
apple1440_df = apple_df.copy(deep=True)

amazon5_df["rounded_time"] = (amazon5_df["published"]).dt.round("5min")
amazon15_df["rounded_time"] = (amazon15_df["published"]).dt.round("15min")
amazon30_df["rounded_time"] = (amazon30_df["published"]).dt.round("30min")
amazon60_df["rounded_time"] = (amazon60_df["published"]).dt.round("1H")
amazon240_df["rounded_time"] = (amazon240_df["published"]).dt.round("4H")
amazon1440_df["rounded_time"] = (amazon1440_df["published"]).dt.round("1D")

apple5_df["rounded_time"] = (apple5_df["published"]).dt.round("5min")
apple15_df["rounded_time"] = (apple15_df["published"]).dt.round("15min")
apple30_df["rounded_time"] = (apple30_df["published"]).dt.round("30min")
apple60_df["rounded_time"] = (apple60_df["published"]).dt.round("1H")
apple240_df["rounded_time"] = (apple240_df["published"]).dt.round("4H")
apple1440_df["rounded_time"] = (apple1440_df["published"]).dt.round("1D")

CHARTS_dir = "./data/CHARTS/"


def join_CHARTS_data_and_GT(source_df, interval, company):
    CHARTS_df = pd.read_csv(
        os.path.join(CHARTS_dir, company + str(interval) + ".csv"),
        header=None,
        index_col=False,
    )
    CHARTS_df.columns = [
        "year.month.day",
        "24hr",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    CHARTS_df["temp_timestamp"] = pd.to_datetime(
        CHARTS_df["year.month.day"].astype(str) + "T" + CHARTS_df["24hr"],
        format="%Y.%m.%dT%H:%M",
    )

    CHARTS_df["label"] = (CHARTS_df["Close"] - CHARTS_df["Open"]).apply(
        lambda x: 1 if x > 0 else -1
    )
    
    right_on = "rounded_time"
    if interval == 60:
        CHARTS_df["timestamp"] = CHARTS_df["temp_timestamp"].dt.round("1H")
    elif interval == 240:
        CHARTS_df["timestamp"] = CHARTS_df["temp_timestamp"].dt.round("4H")
    else:
        CHARTS_df["timestamp"] = CHARTS_df["temp_timestamp"]

    joined_df = pd.merge(
        CHARTS_df,
        source_df,
        left_on="timestamp",
        right_on=right_on,
        how="left",
        suffixes=("", "_y"),
    )
    joined_df.drop(joined_df.filter(regex="_y$").columns.tolist(), axis=1, inplace=True)
    joined_df.drop(["temp_timestamp"], axis=1, inplace=True)

    ########### READ AFTER TESTING
    # joined_df.drop(["entities"], axis=1, inplace=True)
    return joined_df


amazon5_df = join_CHARTS_data_and_GT(amazon5_df, 5, "AMAZON")
amazon15_df = join_CHARTS_data_and_GT(amazon15_df, 15, "AMAZON")
amazon30_df = join_CHARTS_data_and_GT(amazon30_df, 30, "AMAZON")
amazon60_df = join_CHARTS_data_and_GT(amazon60_df, 60, "AMAZON")
amazon240_df = join_CHARTS_data_and_GT(amazon240_df, 240, "AMAZON")
amazon1440_df = join_CHARTS_data_and_GT(amazon1440_df, 1440, "AMAZON")

apple5_df = join_CHARTS_data_and_GT(apple5_df, 5, "APPLE")
apple15_df = join_CHARTS_data_and_GT(apple15_df, 15, "APPLE")
apple30_df = join_CHARTS_data_and_GT(apple30_df, 30, "APPLE")
apple60_df = join_CHARTS_data_and_GT(apple60_df, 60, "APPLE")
apple240_df = join_CHARTS_data_and_GT(apple240_df, 240, "APPLE")
apple1440_df = join_CHARTS_data_and_GT(apple1440_df, 1440, "APPLE")

amazon5_df.dropna(inplace=True)
amazon15_df.dropna(inplace=True)
amazon30_df.dropna(inplace=True)
amazon60_df.dropna(inplace=True)
amazon240_df.dropna(inplace=True)
amazon1440_df.dropna(inplace=True)

apple5_df.dropna(inplace=True)
apple15_df.dropna(inplace=True)
apple30_df.dropna(inplace=True)
apple60_df.dropna(inplace=True)
apple240_df.dropna(inplace=True)
apple1440_df.dropna(inplace=True)

amazon5_df.drop(["text"], axis=1, inplace=True)
amazon15_df.drop(["text"], axis=1, inplace=True)
amazon30_df.drop(["text"], axis=1, inplace=True)
amazon60_df.drop(["text"], axis=1, inplace=True)
amazon240_df.drop(["text"], axis=1, inplace=True)
amazon1440_df.drop(["text"], axis=1, inplace=True)

apple5_df.drop(["text"], axis=1, inplace=True)
apple15_df.drop(["text"], axis=1, inplace=True)
apple30_df.drop(["text"], axis=1, inplace=True)
apple60_df.drop(["text"], axis=1, inplace=True)
apple240_df.drop(["text"], axis=1, inplace=True)
apple1440_df.drop(["text"], axis=1, inplace=True)


def remove_duplicates(df):
    df.sort_values(by=["filteredtext"], inplace=True)
    chunk_size = 1000
    chunks = []
    count = 0
    # come back and take care of boundary conditions
    while not df.empty:
        # print(len(df))
        count += 1
        # print("iteration", count)
        chunks.append(df[:chunk_size])
        df = df.iloc[chunk_size - 1 :]
        if count != 1:
            # print(str(chunks[-1].iloc[-1]["filteredtext"]))
            # print(str(chunks[-2].iloc[0]["filteredtext"]))
            if str(chunks[-1].iloc[-1]["filteredtext"]) == str(
                chunks[-2].iloc[0]["filteredtext"]
            ):
                chunks[-2].drop(chunks[-2].tail(1).index, inplace=True)

    for chunk in chunks:
        chunk["row_string"] = (
            chunk["published"].astype(str)
            + chunk["site"].astype(str)
            + chunk["filteredtext"].astype(str)
        )

        chunk.drop_duplicates(subset=["row_string"], keep="first", inplace=True)
    df = pd.concat(chunks, ignore_index=True)
    df.drop(["row_string"], axis=1, inplace=True)
    # print(len(df))
    return df


amazon5_df = remove_duplicates(amazon5_df)
amazon15_df = remove_duplicates(amazon15_df)
amazon30_df = remove_duplicates(amazon30_df)
amazon60_df = remove_duplicates(amazon60_df)
amazon240_df = remove_duplicates(amazon240_df)
amazon1440_df = remove_duplicates(amazon1440_df)

apple5_df = remove_duplicates(apple5_df)
apple15_df = remove_duplicates(apple15_df)
apple30_df = remove_duplicates(apple30_df)
apple60_df = remove_duplicates(apple60_df)
apple240_df = remove_duplicates(apple240_df)
apple1440_df = remove_duplicates(apple1440_df)

amazon5_df.to_csv("data/amazon5.csv", encoding="utf-8", index=False)
amazon15_df.to_csv("data/amazon15.csv", encoding="utf-8", index=False)
amazon30_df.to_csv("data/amazon30.csv", encoding="utf-8", index=False)
amazon60_df.to_csv("data/amazon60.csv", encoding="utf-8", index=False)
amazon240_df.to_csv("data/amazon240.csv", encoding="utf-8", index=False)
amazon1440_df.to_csv("data/amazon1440.csv", encoding="utf-8", index=False)

apple5_df.to_csv("data/apple5.csv", encoding="utf-8", index=False)
apple15_df.to_csv("data/apple15.csv", encoding="utf-8", index=False)
apple30_df.to_csv("data/apple30.csv", encoding="utf-8", index=False)
apple60_df.to_csv("data/apple60.csv", encoding="utf-8", index=False)
apple240_df.to_csv("data/apple240.csv", encoding="utf-8", index=False)
apple1440_df.to_csv("data/apple1440.csv", encoding="utf-8", index=False)

end = time.time()
print(end - start)
