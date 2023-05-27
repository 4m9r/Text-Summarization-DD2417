import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from transf_model import *
from dataset import *

# load dataset
dataset = SumDataset(max_len=100)

dataset_train, dataset_val, dataset_test = tud.random_split(dataset, [0.8, 0.1, 0.1])

train_loader = tud.DataLoader(dataset_train, batch_size=32, num_workers=8)
val_loader = tud.DataLoader(dataset_val, batch_size=32, num_workers=8)
test_loader = tud.DataLoader(dataset_val, batch_size=32, num_workers=8)


# load model
model = SumTransformer(
    vocab_size=dataset.vocab_size,
    embedding_size=32,
    max_len=dataset.max_len,
    layers=3,
    attention_heads=4,
    hidden_size=32,
    dropout=0.2,
)

model(*list(iter(train_loader))[0])

# logger
logger = pll.TensorBoardLogger("tb_logs", name="summarizer")

# training
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=1,
    logger=logger,
    callbacks=[
        plc.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            verbose=True,
            strict=True,
            min_delta=1e-4,
        )
    ],
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

trainer.test(model, test_loader)

gen = model.inference(
    list(iter(train_loader))[0][0],
    dataset.sos_idx,
    dataset.eos_idx,
    max_len=100,
    mode="greedy",
    device="cpu",
    temperature=1,
)
gen = [dataset.i2w[g] for g in gen]
print(gen)
