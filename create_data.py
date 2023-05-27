import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import string
import re

# get data
df = pd.read_csv("raw/Reviews.csv")
df.drop(
    columns=[
        "Id",
        "ProductId",
        "UserId",
        "ProfileName",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
        "Score",
        "Time",
    ],
    axis=1,
    inplace=True,
)
df = df.dropna()
df = df[:10]

X = list(df["Text"])
Y = list(df["Summary"])

# index
w2i = defaultdict(int)
i2w = defaultdict(str)


def split_line(line):
    line = line.lower()
    for c in string.punctuation:
        line = line.replace(c, "")
    print(line)
    line = line.split(" ")
    line = list(filter(None, line))
    return line


def build_vocab(lines):
    for line in tqdm(lines):
        line = split_line(line)
        for tok in line:
            if not tok in w2i:
                id = len(w2i)
                w2i[tok] = id
                i2w[id] = tok


def index(lines):
    data = []
    for line in tqdm(lines):
        line = split_line(line)
        data.append([w2i[w] for w in line])
    return data


build_vocab(X)
build_vocab(Y)

X = index(X)
Y = index(Y)

data = {"input": X, "target": Y, "w2i": w2i, "i2w": i2w}

with open("data/dataset.json", "w") as f:
    f.write(json.dumps(data))
