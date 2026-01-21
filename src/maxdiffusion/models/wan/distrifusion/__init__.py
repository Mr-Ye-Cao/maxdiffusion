"""
DistriFusion for Wan2.1 in JAX.

DistriFusion key technique:
1. Split sequence across devices: each device processes seq_len/n_devices tokens
2. Self-attention: all-gather KV from all devices, compute attention locally
3. After warmup: use stale KV from previous timestep + async update
4. Cross-attention: cache text KV (doesn't change across timesteps)
5. FFN: token-wise, no communication needed
"""

from .config import DistriFusionConfig
from .attention import (
    DistriFusionSelfAttention,
    DistriFusionCrossAttention,
    KVGatherInterface,
)
from .transformer import DistriFusionTransformerBlock

__all__ = [
    "DistriFusionConfig",
    "DistriFusionSelfAttention",
    "DistriFusionCrossAttention",
    "KVGatherInterface",
    "DistriFusionTransformerBlock",
]
