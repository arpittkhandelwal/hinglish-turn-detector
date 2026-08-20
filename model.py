"""
model.py — WhisperTinyTurnDetector
===================================
Whisper Tiny encoder-only backbone + pooling + 2-layer MLP head.
Supports two pooling modes:
  - 'mean'      : simple mean over time frames
  - 'attention' : single learned query attending over time frames

Design rationale:
  - Whisper Tiny (39M params) was pre-trained on multilingual speech including
    Indic languages (Common Voice Hindi), giving better multilingual representations
    than English-centric models like wav2vec2-base.
  - Encoder-only: decoder adds ~30ms latency + memory without contributing to
    turn detection (we don't need transcription).
  - Attention pooling: a learned query concentrates on prosodically-salient frames
    near the turn boundary rather than averaging over all (including silence) frames.
  - Frozen lower encoder blocks: prevents catastrophic forgetting on a dataset
    that is much smaller and more skewed than Whisper's original pre-training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig


class AttentionPooling(nn.Module):
    """
    Single learned query vector attending over encoder output frames.
    q ∈ R^d, K = V = encoder_output ∈ R^{T x d}
    output = softmax(qK^T / sqrt(d)) @ V  →  R^d
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.scale = hidden_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, D)
        q = self.query.unsqueeze(0).unsqueeze(1)          # (1, 1, D)
        scores = (hidden_states * q).sum(-1) * self.scale  # (B, T)
        weights = F.softmax(scores, dim=-1)                # (B, T)
        pooled = (weights.unsqueeze(-1) * hidden_states).sum(1)  # (B, D)
        return pooled


class WhisperTinyTurnDetector(nn.Module):
    """
    Encoder-only turn detection model based on Whisper Tiny.

    Args:
        freeze_encoder_layers: number of encoder blocks to freeze (from the bottom).
            Whisper Tiny has 4 encoder blocks (0-3). Default: freeze bottom 2.
        pooling: 'mean' or 'attention'.
        dropout: dropout probability for the MLP head.
        hidden_dim: Whisper Tiny encoder hidden size (384).
        mlp_mid_dim: intermediate dimension in MLP head.
    """

    def __init__(
        self,
        freeze_encoder_layers: int = 2,
        pooling: str = "attention",
        dropout: float = 0.15,
        hidden_dim: int = 384,
        mlp_mid_dim: int = 128,
    ):
        super().__init__()
        assert pooling in ("mean", "attention"), "pooling must be 'mean' or 'attention'"

        # Load Whisper Tiny encoder
        model = WhisperModel.from_pretrained("openai/whisper-tiny")
        self.encoder = model.encoder

        # Freeze bottom N encoder blocks
        self._freeze_layers(freeze_encoder_layers)

        # Pooling
        self.pooling_mode = pooling
        if pooling == "attention":
            self.pooling = AttentionPooling(hidden_dim)
        else:
            self.pooling = None  # use torch.mean

        # 2-layer MLP classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mid_dim, 1),
        )

        # Store config info for reporting
        self.freeze_encoder_layers = freeze_encoder_layers
        self.hidden_dim = hidden_dim

    def _freeze_layers(self, n: int):
        """Freeze the first n encoder blocks + the conv feature projection."""
        # Freeze conv layers (always frozen — they are low-level feature extractors)
        for param in self.encoder.conv1.parameters():
            param.requires_grad = False
        for param in self.encoder.conv2.parameters():
            param.requires_grad = False
        for param in self.encoder.embed_positions.parameters():
            param.requires_grad = False

        # Freeze first n transformer blocks
        for i, block in enumerate(self.encoder.layers):
            if i < n:
                for param in block.parameters():
                    param.requires_grad = False

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": round(100.0 * trainable / total, 2),
        }

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (B, 80, T) log-Mel spectrogram from WhisperFeatureExtractor
        Returns:
            logits: (B, 1) raw logits (apply sigmoid for probability)
        """
        encoder_out = self.encoder(input_features)
        hidden = encoder_out.last_hidden_state  # (B, T', D)

        if self.pooling_mode == "attention":
            pooled = self.pooling(hidden)
        else:
            pooled = hidden.mean(dim=1)

        logits = self.head(pooled)  # (B, 1)
        return logits


def build_model(experiment: int) -> WhisperTinyTurnDetector:
    """
    Factory function: return the model config for a given experiment number.

    Experiment 1 — Baseline:
        Mean pooling, all encoder frozen (only head trains)
    Experiment 2 — Attention pool + partial unfreeze:
        Attention pooling, unfreeze top 2 encoder blocks
    Experiment 3 — Same as Exp 2 (hard-negative oversampling is a DATA change):
        Identical architecture to Exp 2; training data differs
    """
    configs = {
        1: dict(freeze_encoder_layers=4, pooling="mean",      dropout=0.15),
        2: dict(freeze_encoder_layers=2, pooling="attention", dropout=0.15),
        3: dict(freeze_encoder_layers=2, pooling="attention", dropout=0.15),
    }
    assert experiment in configs, f"Unknown experiment: {experiment}"
    return WhisperTinyTurnDetector(**configs[experiment])


if __name__ == "__main__":
    # Quick sanity check
    import torch

    for exp in [1, 2, 3]:
        model = build_model(exp)
        params = model.count_params()
        print(f"Experiment {exp}: {params}")

        # Dummy forward pass
        dummy = torch.randn(2, 80, 3000)  # (B=2, mel_bins=80, frames=3000 = ~30s)
        out = model(dummy)
        print(f"  Output shape: {out.shape}")  # expect (2, 1)
