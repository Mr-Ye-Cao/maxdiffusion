"""
DistriFusion configuration for Wan2.1.
"""

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp


@dataclass
class DistriFusionConfig:
    """Configuration for DistriFusion sequence parallelism.

    Attributes:
        n_devices: Number of devices to split sequence across.
        warmup_steps: Number of warmup steps using synchronous all-gather.
            After warmup, uses stale KV + async update.
        mode: Communication mode:
            - "default": Sync during warmup, async after warmup
            - "full_sync": Always use synchronous all-gather
            - "no_sync": No communication (for debugging)
        patch_size: Patch size (t, h, w) for 3D RoPE computation.
        attention_head_dim: Dimension per attention head (for RoPE).
    """
    n_devices: int
    warmup_steps: int = 2
    mode: str = "default"
    patch_size: tuple = (1, 2, 2)
    attention_head_dim: int = 128

    def __post_init__(self):
        assert self.mode in ["default", "full_sync", "no_sync"], \
            f"Invalid mode: {self.mode}"
        assert self.n_devices > 0, f"n_devices must be positive: {self.n_devices}"
        assert self.warmup_steps >= 0, f"warmup_steps must be non-negative: {self.warmup_steps}"


def compute_local_sequence_range(
    full_seq_len: int,
    n_devices: int,
    device_idx: int,
) -> tuple:
    """Compute the start and end indices for local sequence portion.

    Each device processes a contiguous chunk of the sequence.
    The last device handles any remainder.

    Args:
        full_seq_len: Total sequence length.
        n_devices: Number of devices.
        device_idx: Index of current device (0-indexed).

    Returns:
        (start_idx, end_idx): Range of sequence indices for this device.
    """
    local_seq_len = full_seq_len // n_devices
    start_idx = device_idx * local_seq_len

    # Last device takes remainder
    if device_idx == n_devices - 1:
        end_idx = full_seq_len
    else:
        end_idx = start_idx + local_seq_len

    return start_idx, end_idx


def compute_3d_positions_for_local_range(
    start_idx: int,
    end_idx: int,
    grid_sizes: tuple,
) -> tuple:
    """Compute 3D (frame, height, width) positions for local sequence indices.

    For 3D RoPE, we need to convert flat sequence indices to (f, h, w) positions.
    Given grid_sizes = (F, H, W), position i maps to:
        frame_idx = i // (H * W)
        height_idx = (i // W) % H
        width_idx = i % W

    Args:
        start_idx: Start index in full sequence.
        end_idx: End index in full sequence.
        grid_sizes: (num_frames, height, width) after patching.

    Returns:
        (frame_positions, height_positions, width_positions):
            Each is an array of shape (local_len,).
    """
    f, h, w = grid_sizes
    local_len = end_idx - start_idx

    # Create position indices
    pos_indices = jnp.arange(start_idx, end_idx)

    # Convert to 3D positions
    frame_idx = pos_indices // (h * w)
    height_idx = (pos_indices // w) % h
    width_idx = pos_indices % w

    return frame_idx, height_idx, width_idx


def get_local_rotary_emb(
    rotary_emb: jax.Array,
    start_idx: int,
    end_idx: int,
    grid_sizes: tuple,
    attention_head_dim: int,
) -> jax.Array:
    """Get rotary embeddings for local sequence portion with correct 3D positions.

    The full rotary_emb has shape [1, 1, max_seq_len, head_dim//2].
    We need to index it using the 3D positions (frame, height, width).

    For Wan2.1:
    - head_dim = 128
    - Split: t_dim = 128 - 2*(128//6) = 128 - 42 = 86... wait let me check
    - Actually: h_dim = w_dim = 2 * (head_dim // 6) = 2 * 21 = 42
    - t_dim = head_dim - h_dim - w_dim = 128 - 42 - 42 = 44
    - The frequencies are concatenated: [freqs_f, freqs_h, freqs_w]

    Args:
        rotary_emb: Full rotary embeddings [1, 1, seq_len, head_dim//2].
        start_idx: Start index in full sequence.
        end_idx: End index in full sequence.
        grid_sizes: (num_frames, height, width) after patching.
        attention_head_dim: Dimension per attention head.

    Returns:
        local_rotary_emb: Rotary embeddings for local portion.
    """
    f, h, w = grid_sizes
    local_len = end_idx - start_idx

    # Compute split sizes (matching get_frequencies in transformer_wan.py)
    h_dim = w_dim = 2 * (attention_head_dim // 6)
    t_dim = attention_head_dim - h_dim - w_dim

    # Note: The actual rotary_emb in Wan is computed differently
    # It's already the full rotary for each position, we just need to slice
    # For DistriFusion, we need to return the slice for positions [start_idx:end_idx]

    # The rotary_emb is [1, 1, seq_len, head_dim//2] and represents
    # the concatenated frequencies for each position
    # We need positions start_idx to end_idx
    local_rotary_emb = jax.lax.dynamic_slice(
        rotary_emb,
        start_indices=(0, 0, start_idx, 0),
        slice_sizes=(1, 1, local_len, rotary_emb.shape[-1]),
    )

    return local_rotary_emb
