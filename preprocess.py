import os
import glob
import json
import pytz
import nltk

nltk.download("punkt")
import numpy as np
import pandas as pd
import dateutil.parser

# from textblob import TextBlob
from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize
import re
import nltk.data

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
    }
)

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def filter_only_mention(text, stock, company):
    filterlist = []
    sentences = tokenizer.tokenize(text)
    if text.startswith("Official Notice Nr"):
        return None
    for sent in sentences:
        sent = sent.lower()
        sent = sent.replace("\n", " ")
        sent = re.sub(r"[^A-Za-z0-9 \. \:]+", "", sent)
        #         sent = re.sub(' +', ' ', sent)
        #         sent = re.sub('\.+', '.', sent)
        if (stock in sent) or (company in sent):
            filterlist.append(sent)
    if len(filterlist) == 0:
        return None
    return filterlist


amazon_df = pd.DataFrame()
apple_df = pd.DataFrame()

amazon_df = news_df.copy(deep=True)
apple_df = news_df.copy(deep=True)

amazon_df["filteredtext"] = amazon_df["text"].apply(
    filter_only_mention, args=("amzn", "amazon")
)
apple_df["filteredtext"] = apple_df["text"].apply(
    filter_only_mention, args=("aapl", "apple")
)

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
    )
    CHARTS_df.columns = [
        "year.month.day",
        "24hr",
        "Open",
        "Close",
        "High",
        "Low",
        "Volume",
    ]
    CHARTS_df["temp_timestamp"] = pd.to_datetime(
        CHARTS_df["year.month.day"].astype(str) + "T" + CHARTS_df["24hr"],
        format="%Y.%m.%dT%H:%M",
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

    # Adding Ground Truth
    joined_df["label"] = (joined_df["Close"] - joined_df["Open"].shift(1)).apply(
        lambda x: -1 if x < 0 else 1
    )

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

amazon5_df.to_csv("data/amazon5.csv", encoding="utf-8")
amazon15_df.to_csv("data/amazon15.csv", encoding="utf-8")
amazon30_df.to_csv("data/amazon30.csv", encoding="utf-8")
amazon60_df.to_csv("data/amazon60.csv", encoding="utf-8")
amazon240_df.to_csv("data/amazon240.csv", encoding="utf-8")
amazon1440_df.to_csv("data/amazon1440.csv", encoding="utf-8")

apple5_df.to_csv("data/apple5.csv", encoding="utf-8")
apple15_df.to_csv("data/apple15.csv", encoding="utf-8")
apple30_df.to_csv("data/apple30.csv", encoding="utf-8")
apple60_df.to_csv("data/apple60.csv", encoding="utf-8")
apple240_df.to_csv("data/apple240.csv", encoding="utf-8")
apple1440_df.to_csv("data/apple1440.csv", encoding="utf-8")
