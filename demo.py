import torch
from unidecode import unidecode
import string
import torch.nn as nn
import torch.utils.data as tud
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
import sys

from transf_model import *
from dataset import *
import random

with open("data/char_dataset.json", "r") as f:
    data = json.loads(f.read())

w2i = data["w2i"]
i2w = data["i2w"]

# padding, sos, eos
id = len(w2i)
w2i["<sos>"] = id
i2w[str(id)] = "<sos>"

id = len(w2i)
w2i["<eos>"] = id
i2w[str(id)] = "<eos>"

id = len(w2i)
w2i["<pad>"] = id
i2w[str(id)] = "<pad>"


# load model
model = SumTransformer.load_from_checkpoint(
    "tb_logs/new_char_summarizer/version_1/checkpoints/epoch=6-step=84574.ckpt"
)
model.eval()


def split_line(line):
    line = unidecode(line)
    line = line.lower()
    for c in string.punctuation:
        line = line.replace(c, "")
    line = list(line)
    line = list(filter(None, line))
    return line


def process(X, max_len):
    X = X[: max_len - 2]
    return F.pad(
        torch.LongTensor([w2i["<sos>"]] + X + [w2i["<eos>"]]),
        (0, max_len - len(X) - 2),
        mode="constant",
        value=w2i["<pad>"],
    )


input_text = sys.argv[1]

print(f"Test sample:\n", input_text)

gen = model.inference(
    process([w2i[c] for c in split_line(input_text)], 256),
    w2i["<sos>"],
    w2i["<eos>"],
    max_len=100,
    mode="topp",
    temperature=1,
)

summary = [i2w[str(g)] for g in gen]
print("Generated summary:\n", "".join(summary))
