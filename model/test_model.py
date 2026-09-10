"""Smoke test: load data, build the model, run one forward pass, print shapes.

Run this before training. If it prints shapes and finishes, the pieces fit.
"""

import torch
from torch.utils.data import DataLoader

from data import WaveformDataset
from model import WaveformTransformer

BATCH_SIZE = 8

dataset = WaveformDataset()
num_traces, num_samples = dataset.voltage.shape
print(f"dataset: {num_traces} traces x {num_samples} samples, "
      f"{dataset.parameters.shape[1]} parameters each")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
parameters, voltage = next(iter(loader))
print(f"batch:   parameters {tuple(parameters.shape)}, voltage {tuple(voltage.shape)}")

model = WaveformTransformer()
num_weights = sum(p.numel() for p in model.parameters())
print(f"model:   {num_weights:,} trainable weights")

# Teacher-forced forward pass (what train.py will call).
predicted, hidden = model(parameters, voltage)
print(f"forward: predicted {tuple(predicted.shape)}, hidden {tuple(hidden.shape)}")

# An untrained model should give a loss near 1.0 because the targets are
# normalised to unit variance; wildly different means normalisation is broken.
loss = torch.nn.functional.mse_loss(predicted, voltage)
print(f"loss:    {loss.item():.3f} (untrained, expect ~1)")

# Attention maps are stashed on each attention module after a forward pass.
attention = model.blocks[0].attention.last_attention
print(f"attn:    layer 0 attention {tuple(attention.shape)} (batch, heads, tokens, tokens)")

# Autoregressive generation from parameters alone (what generate.py will call).
model.eval()
generated = model.generate(parameters[:2], num_samples)
print(f"generate: {tuple(generated.shape)} from 2 parameter vectors")

print("ok")
