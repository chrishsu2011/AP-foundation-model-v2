"""A small decoder-only transformer that generates HH voltage traces patch by patch.

Design summary
--------------
Tokenisation : PATCH embeddings. The trace is cut into non-overlapping windows
               of PATCH_SIZE samples and each window becomes one token via a
               linear layer. Raw per-sample tokens would make the sequence 5000
               long (too slow, too easy to overfit); binning the voltage would
               throw away amplitude precision that the spike peak depends on.
               Patches keep the trace continuous and shrink 5000 samples to
               100 tokens.
Conditioning : PREFIX token. The 26 parameters are projected to one vector and
               placed at position 0, in front of the patches. Every later token
               can attend to it, and its attention weights are directly
               visible — useful later when we ask which parts of the waveform
               "look at" the biophysics.
Architecture : Decoder-only, causal (each token sees only earlier tokens), so
               the same model both trains with teacher forcing and generates
               autoregressively. Pre-LayerNorm blocks, learned positions.
Size         : 3 layers x 64 hidden x 4 heads ~ 150k parameters. Small on
               purpose: 1000 traces is a tiny dataset.
"""

import math

import torch
import torch.nn as nn

PATCH_SIZE = 50        # samples per token: 50 * 0.01 ms = 0.5 ms of signal
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
NUM_PARAMETERS = 26
DROPOUT = 0.1
MAX_TOKENS = 128       # 1 parameter token + up to 127 patches (5000 / 50 = 100)


class CausalSelfAttention(nn.Module):
    """Multi-head attention where token i may only look at tokens <= i."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # One linear layer produces queries, keys and values together; it is
        # cheaper than three separate layers and mathematically identical.
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Filled in on every forward pass so attention maps can be inspected
        # after the fact without changing the model's return values.
        self.last_attention = None

    def forward(self, x):
        batch, seq_len, hidden_dim = x.shape

        query, key, value = self.qkv(x).split(hidden_dim, dim=-1)
        # (batch, seq, hidden) -> (batch, heads, seq, head_dim) so each head
        # attends independently over its own slice of the hidden vector.
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale by sqrt(head_dim) so the dot products do not grow with head
        # size and push softmax into a regime with vanishing gradients.
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Upper-triangular mask blocks attention to future tokens. Without it
        # the model could copy the next patch straight from the input during
        # training and would learn nothing usable for generation.
        future = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float("-inf"))

        attention = scores.softmax(dim=-1)
        self.last_attention = attention.detach()  # (batch, heads, seq, seq)
        attention = self.dropout(attention)

        mixed = attention @ value                                   # (batch, heads, seq, head_dim)
        mixed = mixed.transpose(1, 2).reshape(batch, seq_len, hidden_dim)
        return self.output(mixed)


class TransformerBlock(nn.Module):
    """Attention + MLP, each wrapped in a residual connection."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        # The 4x expansion is the standard transformer MLP width; it gives the
        # block somewhere to do per-token computation between attention steps.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm (normalise BEFORE each sub-layer) trains more stably than
        # post-norm for small models and needs no warm-up schedule.
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class WaveformTransformer(nn.Module):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_parameters=NUM_PARAMETERS,
        dropout=DROPOUT,
        max_tokens=MAX_TOKENS,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.parameter_embedding = nn.Linear(num_parameters, hidden_dim)
        self.patch_embedding = nn.Linear(patch_size, hidden_dim)
        # Learned positions rather than sinusoidal: the sequence length is
        # fixed and short, so there is nothing to extrapolate to.
        self.position_embedding = nn.Embedding(max_tokens, hidden_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        # Each output token is decoded straight back into PATCH_SIZE voltage
        # samples: plain regression, no vocabulary, no sampling temperature.
        self.patch_head = nn.Linear(hidden_dim, patch_size)

    def run_blocks(self, parameters, patches):
        """Shared by forward() and generate(). Returns hidden states (batch, tokens, hidden)."""
        batch = parameters.shape[0]

        parameter_token = self.parameter_embedding(parameters).unsqueeze(1)  # (batch, 1, hidden)
        patch_tokens = self.patch_embedding(patches)                         # (batch, n, hidden)
        tokens = torch.cat([parameter_token, patch_tokens], dim=1)

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.position_embedding(positions)

        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)

    def forward(self, parameters, voltage):
        """Teacher-forced pass for training.

        parameters : (batch, 26)  normalised
        voltage    : (batch, T)   normalised, T must be divisible by patch_size
        returns    : predicted voltage (batch, T), hidden states (batch, 1 + T/patch_size, hidden)
        """
        batch, num_samples = voltage.shape
        assert num_samples % self.patch_size == 0, (
            f"trace length {num_samples} is not a multiple of patch_size {self.patch_size}"
        )
        patches = voltage.view(batch, num_samples // self.patch_size, self.patch_size)

        # Input is [params, patch_0, ..., patch_{n-2}]; the output at each
        # position predicts the NEXT patch. So the parameter token alone must
        # predict patch_0 — the first 0.5 ms is determined purely by the biophysics.
        hidden = self.run_blocks(parameters, patches[:, :-1])
        predicted = self.patch_head(hidden)                    # (batch, n, patch_size)
        return predicted.reshape(batch, num_samples), hidden

    @torch.no_grad()
    def generate(self, parameters, num_samples):
        """Autoregressively produce a trace of num_samples points from parameters alone."""
        batch = parameters.shape[0]
        num_patches = num_samples // self.patch_size
        patches = torch.zeros(batch, 0, self.patch_size, device=parameters.device)

        for _ in range(num_patches):
            hidden = self.run_blocks(parameters, patches)
            # Only the last position's output is new; earlier ones were already
            # consumed. Recomputing them is wasteful but keeps the code short.
            next_patch = self.patch_head(hidden[:, -1:])
            patches = torch.cat([patches, next_patch], dim=1)

        return patches.reshape(batch, num_patches * self.patch_size)
