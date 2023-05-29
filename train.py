import torch
import torch.nn as nn
import torch.utils.data as tud
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll

from transf_model import *
from dataset import *

torch.multiprocessing.set_sharing_strategy("file_system")
torch.manual_seed(0)

# load dataset
dataset = SumDataset(text_max_len=256, summary_max_len=128, char_level=True)

dataset_train, dataset_val, dataset_test = tud.random_split(dataset, [0.8, 0.1, 0.1])

batch_size = 16
train_loader = tud.DataLoader(dataset_train, batch_size=batch_size, num_workers=8)
val_loader = tud.DataLoader(dataset_val, batch_size=batch_size, num_workers=8)
test_loader = tud.DataLoader(dataset_test, batch_size=batch_size, num_workers=8)


# load model
model = SumTransformer(
    vocab_size=dataset.vocab_size,
    embedding_size=64,
    layers=3,
    attention_heads=4,
    hidden_size=64,
    dropout=0.2,
    pad_index=dataset.pad_idx,
)

# logger
logger = pll.TensorBoardLogger("tb_logs", name="char_summarizer")

# training
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    logger=logger,
    callbacks=[
        # Early stopping
        plc.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            verbose=True,
            strict=True,
            min_delta=1e-4,
        ),
        # saves top-K checkpoints based on "val_loss" metric
        plc.ModelCheckpoint(
            save_top_k=2,
            monitor="val_loss",
            mode="min",
        ),
    ],
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

trainer.test(model, test_loader)
