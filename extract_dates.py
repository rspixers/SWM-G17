import glob
import json
import os
import pandas as pd
from datetime import datetime, timedelta


def join_CHARTS_data(interval, company):
    CHARTS_df = pd.read_csv(
        os.path.join(CHARTS_dir, "CHARTS/" + company + str(interval) + ".csv"),
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
    CHARTS_df["timestamp"] = pd.to_datetime(
        CHARTS_df["year.month.day"].astype(str) + "T" + CHARTS_df["24hr"],
        format="%Y.%m.%dT%H:%M",
    )
    print(CHARTS_df)
    right_on = "published_" + str(interval)
    joined_df = pd.merge(
        CHARTS_df, news_df, left_on="timestamp", right_on=right_on, how="left"
    )
    ########### READD AFTER TESTING
    joined_df.drop(["entities"], axis=1, inplace=True)
    return joined_df


json_dir = r"ENTER_DIR_HERE"
count = 0

sub_folders = os.walk(json_dir)
data = []
for folder_index, json_folder_contents in enumerate(sub_folders):
    if folder_index != 0:
        json_files = json_folder_contents[2]
        for json_file in json_files:
            json_filepath = os.path.join(json_folder_contents[0], json_file)
            with open(json_filepath, encoding="utf-8") as json_data:
                data.append(json.load(json_data))
            count = count + 1
        print(folder_index)
print(count)


for record in data:
    local_time, offset = record["published"].split("+")
    local_datetime = datetime.strptime(local_time, "%Y-%m-%dT%H:%M:%S.%f")
    offset_datetime = datetime.strptime(offset, "%H:%M")
    record["published"] = local_datetime - timedelta(
        hours=offset_datetime.hour, minutes=offset_datetime.minute
    )


news_df = pd.DataFrame.from_dict(data, orient="columns")
news_df["published_5"] = news_df["published"].dt.round("5min")
news_df["published_15"] = news_df["published"].dt.round("15min")
news_df["published_30"] = news_df["published"].dt.round("30min")
news_df["published_60"] = news_df["published"].dt.round("1H")
news_df["published_240"] = news_df["published"].dt.round("4H")
news_df["published_1440"] = news_df["published"].dt.round("1D")

CHARTS_dir = r"ENTER_DIR_HERE"

amazon5_df = join_CHARTS_data(5, "AMAZON")
amazon15_df = join_CHARTS_data(15, "AMAZON")
amazon30_df = join_CHARTS_data(30, "AMAZON")
amazon60_df = join_CHARTS_data(60, "AMAZON")
amazon240_df = join_CHARTS_data(240, "AMAZON")
amazon1440_df = join_CHARTS_data(1440, "AMAZON")
apple5_df = join_CHARTS_data(5, "APPLE")
apple15_df = join_CHARTS_data(15, "APPLE")
apple30_df = join_CHARTS_data(30, "APPLE")
apple60_df = join_CHARTS_data(60, "APPLE")
apple240_df = join_CHARTS_data(240, "APPLE")
apple1440_df = join_CHARTS_data(1440, "APPLE")

print(amazon5_df["timestamp"])
print(amazon5_df["text"][1842])
# Outputting data for visual QA Testing / writing to and from csv's should be discouraged since the csv's will most likely not handle text well if delimiter is seen
# amazon5_df.to_csv("data/amazon5.csv", index=False, encoding="utf-8")
# amazon15_df.to_csv("data/amazon15.csv", index=False, encoding="utf-8")
# amazon30_df.to_csv("data/amazon30.csv", index=False, encoding="utf-8")
# amazon60_df.to_csv("data/amazon60.csv", index=False, encoding="utf-8")
# amazon240_df.to_csv("data/amazon240.csv", index=False, encoding="utf-8")
# amazon1440_df.to_csv("data/amazon1440.csv", index=False, encoding="utf-8")
# apple5_df.to_csv("data/apple5.csv", index=False, encoding="utf-8")
# apple15_df.to_csv("data/apple15.csv", index=False, encoding="utf-8")
# apple30_df.to_csv("data/apple30.csv", index=False, encoding="utf-8")
# apple60_df.to_csv("data/apple60.csv", index=False, encoding="utf-8")
# apple240_df.to_csv("data/apple240.csv", index=False, encoding="utf-8")
# apple1440_df.to_csv("data/apple1440.csv", index=False, encoding="utf-8")
