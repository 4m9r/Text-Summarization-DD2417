import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from transf_model import *
from dataset import *
import random

torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)
random.seed(0)

# load dataset
dataset = SumDataset(text_max_len=256, summary_max_len=128, char_level=True)

dataset_train, _, dataset_test = tud.random_split(dataset, [0.8, 0.1, 0.1])

# load model
model = SumTransformer.load_from_checkpoint(
    "tb_logs/char_summarizer/version_3/checkpoints/epoch=10-step=132902.ckpt"
)
model.eval()

x, y, z = dataset[0]
print("".join([dataset.i2w[str(c.item())] for c in x]))
print("".join([dataset.i2w[str(c.item())] for c in y]))
print("".join([dataset.i2w[str(c.item())] for c in z]))

"""
s = ""
for i in range(10):
    index = random.randint(0, len(dataset_test))

    input_text = dataset_test[index][0]
    original_sum = dataset_test[index][1]

    text = [dataset.i2w[str(g.item())] for g in input_text]
    true_summary = [dataset.i2w[str(g.item())] for g in original_sum]
    print(f"Test sample {index}:\n", "".join(text))
    print("Original summary:\n", "".join(true_summary))
    s += f"Test sample {index}:\n" + "".join(text)
    s += "\nOriginal summary:\n" + "".join(true_summary)

    gen = model.inference(
        input_text,
        dataset.sos_idx,
        dataset.eos_idx,
        max_len=100,
        mode="topk",
        temperature=1,
    )

    summary = [dataset.i2w[str(g)] for g in gen]
    print("Generated summary:\n", "".join(summary))
    s += "\nGenerated summary:\n" + "".join(summary)
    s += "\n\n"

with open("results_topk.txt", "w") as f:
    f.write(s)
"""
