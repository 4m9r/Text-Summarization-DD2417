import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from torch.nn import functional as F


# from pytorch docs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.swapaxes(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        x = x.swapaxes(0, 1)
        return x


class SumTransformer(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        layers: int,
        attention_heads: int,
        hidden_size: int,
        dropout: float,
        pad_index: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.layers = layers
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.positional_encoding = PositionalEncoding(
            d_model=embedding_size, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=layers,
        )

        self.prediction_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.save_hyperparameters()

    # (B, L)
    def forward(self, X, Y):
        # (B, L, E)
        embedded_input = self.embedding(X)

        # (B, L, E)
        embedded_input = self.positional_encoding(embedded_input)

        # encode
        mask = nn.Transformer.generate_square_subsequent_mask(
            X.shape[1], device=X.device
        ).bool()

        latent = self.encoder(embedded_input, mask=mask)

        # decode
        embedded_target = self.embedding(Y)
        embedded_target = self.positional_encoding(embedded_target)

        mask = nn.Transformer.generate_square_subsequent_mask(
            Y.shape[1], device=X.device
        ).bool()
        # (B, L, H)
        transf_out = self.decoder(tgt=embedded_target, memory=latent, tgt_mask=mask)

        # (B, L, V)
        logits = self.prediction_layer(transf_out)

        return logits

    def training_step(self, train_batch, batch_idx):
        x, y, z = train_batch
        x_hat = self.forward(x, y).swapaxes(1, 2)
        loss = self.loss(x_hat, z)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, z = val_batch
        x_hat = self.forward(x, y).swapaxes(1, 2)
        loss = self.loss(x_hat, z)
        self.log("val_loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y, z = test_batch
        x_hat = self.forward(x, y).swapaxes(1, 2)
        loss = self.loss(x_hat, z)
        self.log("test_loss", loss)
        return loss

    def inference(
        self,
        X,
        sos_idx,
        eos_idx,
        max_len=100,
        mode="greedy",
        temperature=1,
    ):
        X = X.to(self.device)
        X = X.unsqueeze(0)
        embedded_input = self.embedding(X)
        embedded_input = self.positional_encoding(embedded_input)

        mask = nn.Transformer.generate_square_subsequent_mask(
            X.shape[1], device=self.device
        ).to(torch.bool)

        transf_encoder_out = self.encoder(embedded_input, mask=mask)

        with torch.no_grad():
            generation = [sos_idx]
            t = 0
            while t < max_len:
                input = torch.LongTensor(generation).to(self.device).unsqueeze(0)
                embedded_input = self.embedding(input)
                embedded_input = self.positional_encoding(embedded_input)

                mask = nn.Transformer.generate_square_subsequent_mask(
                    len(generation), device=self.device
                ).to(torch.bool)

                out = self.decoder(
                    tgt=embedded_input, memory=transf_encoder_out, tgt_mask=mask
                )
                logits = self.prediction_layer(out)[:, -1, :]

                tok = self.sample(logits, mode=mode, T=temperature)
                generation.append(tok)

                if tok == eos_idx:
                    break
                t += 1

            return generation

    def sample(self, out, mode="greedy", K=5, T=1, P=0.9):
        if mode == "greedy":
            sample = torch.argmax(out)

        elif mode == "topk":
            values, indexes = torch.topk(out, K, dim=-1)
            out = out.clone().squeeze(1)
            out[out < values[:, -1]] = -float("Inf")
            probs = self.softmax(out / T).squeeze()
            sample = torch.multinomial(probs, 1)

        elif mode == "topp":
            values, indexes = torch.sort(out / T, descending=True)
            values = self.softmax(values)
            cum_probs = torch.cumsum(values, dim=-1)

            remove = cum_probs > P
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0

            out = out.clone()
            remove = torch.zeros_like(out, dtype=torch.bool).scatter_(
                dim=-1, index=indexes, src=remove
            )
            out[remove] = -float("Inf")

            probs = self.softmax(out / T).squeeze()
            sample = torch.multinomial(probs, 1)

        return sample.item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
