import torch
from model.model_mae import WaveformMAE
from model.data import WaveformDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

PATCH_SIZE = 50        # samples per patch: 50 * 0.01 ms = 0.5 ms of signal
ENCODER_DIM = 64       # hidden dimension of the encoder transformer
ENCODER_HEADS = 4      # attention heads in encoder
ENCODER_LAYERS = 3     # depth of encoder
DECODER_DIM = 32       # decoder is intentionally smaller — it's just a training tool
DECODER_HEADS = 4
DECODER_LAYERS = 1
DROPOUT = 0.1
MAX_PATCHES = 128      # max sequence length (128 * 50 = 6400 samples)
MASK_RATIO = 0.5   
GRADIENT_CLIP = 1
BATCH_SIZE = 64

import os
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "waveforms", "hh_dataset_1ms.h5")

dataset = WaveformDataset(filepath)
train_set, val_set = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print(torch.__version__)
print(torch.cuda.is_available())

model = WaveformMAE(
    patch_size=PATCH_SIZE,        # 50
    encoder_dim=ENCODER_DIM,      # 64
    encoder_heads=ENCODER_HEADS,  # 4
    encoder_layers=ENCODER_LAYERS,# 3
    decoder_dim=DECODER_DIM,      # 32
    decoder_heads=DECODER_HEADS,  # 4
    decoder_layers=DECODER_LAYERS,# 1
    dropout=DROPOUT,              # 0.1
    max_patches=MAX_PATCHES,      # 128
)

# crashes if not using cuda
assert torch.cuda.is_available()
device = torch.device("cuda")

print("Using: ", device)

model = model.to(device)

# ---

# where loader is the DataLoader object and training is a boolean for toggling between training and test/validation
def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    num_batches = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for parameters, voltage in loader:        # MAE ignores parameters
            voltage = voltage.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred_patches, mask = model(voltage, mask_ratio=MASK_RATIO)
            target_patches = model.patchify(voltage)
            per_patch_mse = ((pred_patches - target_patches) ** 2).mean(dim=-1)
            loss = (per_patch_mse * mask).sum() / mask.sum()

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---

num_epochs = 50

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:>3}/{num_epochs} | MSE: {train_loss:.6f}")

torch.save({
"model": model.state_dict(),
"optimizer": optimizer.state_dict(),
"epoch": num_epochs,
"train_losses": train_losses,
"val_losses": val_losses,

"config": {
        "patch_size": PATCH_SIZE,
        "encoder_dim": ENCODER_DIM,
        "encoder_heads": ENCODER_HEADS,
        "encoder_layers": ENCODER_LAYERS,
        "decoder_dim": DECODER_DIM,
        "decoder_heads": DECODER_HEADS,
        "decoder_layers": DECODER_LAYERS,
        "dropout": DROPOUT,
        "max_patches": MAX_PATCHES,
        "mask_ratio": MASK_RATIO,
        "lr": 1e-4,
        "batch_size": BATCH_SIZE,
    },

"voltage_mean": dataset.voltage_mean,
"voltage_std": dataset.voltage_std,
}, "mae_checkpoint.pt")