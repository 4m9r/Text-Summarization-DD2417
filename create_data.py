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
# df = df[:10000]

X = list(df["Text"])
Y = list(df["Summary"])

# index
w2i = defaultdict(None)
i2w = defaultdict(None)

max_len = 30


def split_line(line):
    line = line.lower()
    for c in string.punctuation:
        line = line.replace(c, "")
    line = line.split(" ")
    line = list(filter(None, line))
    if len(line) > max_len:
        return None
    return line


def build_vocab(lines):
    for line in tqdm(lines):
        line = split_line(line)
        if line is not None:
            for tok in line:
                if not tok in w2i:
                    id = len(w2i)
                    w2i[tok] = id
                    i2w[id] = tok


def index(lines):
    data = []
    for line in tqdm(lines):
        line = split_line(line)
        if line is not None:
            data.append([w2i[w] for w in line])
    return data


build_vocab(X)
build_vocab(Y)

X = index(X)
Y = index(Y)

data = {"input": X, "target": Y, "w2i": w2i, "i2w": i2w}

with open("data/dataset.json", "w") as f:
    f.write(json.dumps(data))

print("Total number of words: ", len(w2i))
print("Total number of datapoints: ", len(X))
