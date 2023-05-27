import torch
import torch.utils.data as tud
import torch.nn.functional as F
from tqdm import tqdm
import json


class SumDataset(tud.Dataset):
    def __init__(self, max_len):
        with open("data/dataset.json", "r") as f:
            data = json.loads(f.read())

        self.w2i = data["w2i"]
        self.i2w = data["i2w"]

        # padding, sos, eos
        id = len(self.w2i)
        self.w2i["<sos>"] = id
        self.i2w[id] = "<sos>"

        id = len(self.w2i)
        self.w2i["<eos>"] = id
        self.i2w[id] = "<eos>"

        id = len(self.w2i)
        self.w2i["<pad>"] = id
        self.i2w[id] = "<pad>"

        self.vocab_size = len(self.w2i)
        self.max_len = max_len

        self.inputs = []
        self.targets = []

        for x, y in tqdm(zip(data["input"], data["target"])):
            x = self.process(x)
            y = self.process(y)
            self.inputs.append(x)
            self.targets.append(y)

    def process(self, X):
        X = X[: self.max_len - 2]
        return F.pad(
            torch.LongTensor([self.w2i["<sos>"]] + X + [self.w2i["<eos>"]]),
            (0, self.max_len - len(X) - 2),
            mode="constant",
            value=self.w2i["<pad>"],
        )

    @property
    def sos_idx(self):
        return self.w2i["<sos>"]

    @property
    def eos_idx(self):
        return self.w2i["<eos>"]

    @property
    def pad_idx(self):
        return self.w2i["<pad>"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.targets[idx])
