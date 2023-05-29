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

source_X = list(df["Text"])
source_Y = list(df["Summary"])

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

X = []
Y = []

for x, y in zip(source_X, source_Y):
    line_x = split_line(x)
    line_y = split_line(y)
    if len(line_x) < max_len and len(line_y) < max_len:
        X.append(x)
        Y.append(y)

print(X[0], Y[0])

build_vocab(X)
build_vocab(Y)
X = index(X)
Y = index(Y)

print("Max sequence in X: ", max([len(x) for x in X]))
print("Max sequence in Y: ", max([len(y) for y in Y]))

data = {"input": X, "target": Y, "w2i": w2i, "i2w": i2w}

with open("data/char_dataset.json", "w") as f:
    f.write(json.dumps(data))

print("Total number of words: ", len(w2i))
print("Total number of datapoints: ", len(X))
