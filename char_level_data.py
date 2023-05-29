import pandas as pd
import json
import string
import re
from tqdm import tqdm
from collections import defaultdict
from unidecode import unidecode

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


def split_line(line):
    line = unidecode(line)
    line = line.lower()
    for c in string.punctuation:
        line = line.replace(c, "")
    line = list(line)
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


max_len = 250

X = [
    x
    for x, y in zip(X, Y)
    if len(split_line(x)) < max_len and len(split_line(y)) < max_len
]
Y = [
    y
    for x, y in zip(X, Y)
    if len(split_line(x)) < max_len and len(split_line(y)) < max_len
]

build_vocab(X)
build_vocab(Y)

X = index(X)
Y = index(Y)

print("Max sequence in X: ", max([len(x) for x in X]))
print("Max sequence in Y: ", max([len(y) for y in Y]))

data = {"input": X, "target": Y, "w2i": w2i, "i2w": i2w}

with open("data/char_dataset.json", "w") as f:
    f.write(json.dumps(data))

print(w2i.keys())
print("Total number of words: ", len(w2i))
print("Total number of datapoints: ", len(X))
