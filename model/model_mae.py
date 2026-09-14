"""Masked autoencoder (MAE) for unsupervised learning on HH voltage traces.

Design summary
--------------
Objective  : Self-supervised. Mask random patches of the waveform and
             reconstruct them from visible context. No labels, no parameter
             conditioning — the model learns waveform structure alone.
Encoder    : Bidirectional transformer that processes ONLY visible patches.
             After training, run the encoder on ALL patches (no masking)
             to extract embeddings for downstream probing.
Decoder    : Smaller transformer that takes encoded visible patches plus
             learned mask tokens, restores original ordering, and predicts
             the masked patches. Disposable after training.
Masking    : 75% of patches masked by default. Aggressive masking forces
             the encoder to learn global structure (spike shape, timing,
             recovery dynamics) rather than local interpolation.
Reference  : He et al., "Masked Autoencoders Are Scalable Vision Learners",
             CVPR 2022 — adapted from 2D image patches to 1D time series.
"""

import torch
import torch.nn as nn

# --- Defaults -----------------------------------------------------------------
PATCH_SIZE = 50        # samples per patch: 50 * 0.01 ms = 0.5 ms of signal
ENCODER_DIM = 64       # hidden dimension of the encoder transformer
ENCODER_HEADS = 4      # attention heads in encoder
ENCODER_LAYERS = 3     # depth of encoder
DECODER_DIM = 32       # decoder is intentionally smaller — it's just a training tool
DECODER_HEADS = 4
DECODER_LAYERS = 1
DROPOUT = 0.1
MAX_PATCHES = 128      # max sequence length (128 * 50 = 6400 samples)
MASK_RATIO = 0.75      # fraction of patches to mask during training


# --- Building blocks ----------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention, bidirectional (no causal mask)."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Stored after every forward pass so attention maps can be inspected
        # without modifying the return signature.
        self.last_attention = None

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)

        reshape = lambda t: t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = scores.softmax(dim=-1)
        self.last_attention = attn.detach()       # (B, heads, S, S)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.output(out)


class TransformerBlock(nn.Module):
    """Pre-norm attention + feed-forward with residual connections."""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --- Main model ---------------------------------------------------------------

class WaveformMAE(nn.Module):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        encoder_dim=ENCODER_DIM,
        encoder_heads=ENCODER_HEADS,
        encoder_layers=ENCODER_LAYERS,
        decoder_dim=DECODER_DIM,
        decoder_heads=DECODER_HEADS,
        decoder_layers=DECODER_LAYERS,
        dropout=DROPOUT,
        max_patches=MAX_PATCHES,
    ):
        super().__init__()
        self.patch_size = patch_size

        # --- Encoder ---
        self.patch_embed = nn.Linear(patch_size, encoder_dim)
        # Learned positional embeddings. The sequence length is fixed and short,
        # so sinusoidal offers no advantage.
        self.encoder_pos = nn.Embedding(max_patches, encoder_dim)
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(encoder_dim, encoder_heads, dropout)
             for _ in range(encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # --- Decoder ---
        # Project encoder output to the (smaller) decoder space.
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        # A single learned vector that stands in for every masked patch.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos = nn.Embedding(max_patches, decoder_dim)
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, decoder_heads, dropout)
             for _ in range(decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Decode each token back to raw voltage samples.
        self.decoder_head = nn.Linear(decoder_dim, patch_size)

    # --- Helpers ---------------------------------------------------------------

    def patchify(self, voltage):
        """(B, T) -> (B, num_patches, patch_size)"""
        B, T = voltage.shape
        assert T % self.patch_size == 0, (
            f"trace length {T} not divisible by patch_size {self.patch_size}"
        )
        return voltage.view(B, T // self.patch_size, self.patch_size)

    def random_masking(self, tokens, mask_ratio):
        """Mask a random subset of patches using the noise-sort trick from MAE.

        Returns
        -------
        visible      : (B, num_keep, D) tokens that survived masking
        mask         : (B, N) binary, 1 = masked, 0 = visible
        ids_restore  : (B, N) indices to un-shuffle visible + mask tokens
                       back to original patch order
        ids_keep     : (B, num_keep) which positions survived
        """
        B, N, D = tokens.shape
        num_keep = int(N * (1 - mask_ratio))

        # Each patch gets random noise; sort ascending; keep the quietest ones.
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        visible = torch.gather(
            tokens, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        # Build the binary mask in the original (un-shuffled) ordering.
        mask = torch.ones(B, N, device=tokens.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible, mask, ids_restore, ids_keep

    # --- Encode / Decode -------------------------------------------------------

    def encode(self, voltage, mask_ratio=None):
        """Embed and encode patches.

        mask_ratio=None  → encode ALL patches (for probing after training).
        mask_ratio=0.75  → encode only visible patches (for training).

        Returns (encoded, mask, ids_restore, ids_keep).
        mask/ids are None when no masking is applied.
        """
        patches = self.patchify(voltage)          # (B, N, patch_size)
        tokens = self.patch_embed(patches)        # (B, N, encoder_dim)

        if mask_ratio is not None and mask_ratio > 0:
            tokens, mask, ids_restore, ids_keep = self.random_masking(
                tokens, mask_ratio
            )
            # Add position embeddings for the KEPT positions only.
            tokens = tokens + self.encoder_pos(ids_keep)
        else:
            pos = torch.arange(tokens.shape[1], device=voltage.device)
            tokens = tokens + self.encoder_pos(pos)
            mask, ids_restore, ids_keep = None, None, None

        for block in self.encoder_blocks:
            tokens = block(tokens)

        return self.encoder_norm(tokens), mask, ids_restore, ids_keep

    def decode(self, encoded, ids_restore, num_patches):
        """Reconstruct all patches from encoded visible patches + mask tokens."""
        B = encoded.shape[0]
        visible = self.enc_to_dec(encoded)

        # Fill the masked slots with copies of the learned mask token.
        num_masked = num_patches - visible.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)

        # Concatenate, then un-shuffle to restore original temporal order.
        full = torch.cat([visible, mask_tokens], dim=1)
        full = torch.gather(
            full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1]),
        )

        pos = torch.arange(num_patches, device=full.device)
        full = full + self.decoder_pos(pos)

        for block in self.decoder_blocks:
            full = block(full)

        return self.decoder_head(self.decoder_norm(full))  # (B, N, patch_size)

    # --- Public API ------------------------------------------------------------

    def forward(self, voltage, mask_ratio=MASK_RATIO):
        """Training forward pass.

        Parameters
        ----------
        voltage    : (B, T) normalised voltage trace
        mask_ratio : fraction of patches to mask

        Returns
        -------
        pred : (B, N, patch_size) reconstructed patches for ALL positions
        mask : (B, N) binary mask — 1 = was masked, 0 = was visible
        """
        num_patches = voltage.shape[1] // self.patch_size

        encoded, mask, ids_restore, _ = self.encode(voltage, mask_ratio)
        pred = self.decode(encoded, ids_restore, num_patches)

        return pred, mask

    @torch.no_grad()
    def get_embeddings(self, voltage):
        """Encode all patches (no masking) → per-patch embeddings.

        For probing, mean-pool over the patch dimension:
            emb = model.get_embeddings(v).mean(dim=1)   # (B, encoder_dim)
        """
        encoded, _, _, _ = self.encode(voltage, mask_ratio=None)
        return encoded   # (B, num_patches, encoder_dim)
