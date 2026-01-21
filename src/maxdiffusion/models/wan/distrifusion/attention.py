"""
DistriFusion attention wrappers for Wan2.1.

Key techniques:
1. Self-attention: all-gather KV from all devices before attention
2. After warmup: use stale KV from previous timestep + async update
3. Cross-attention: cache text KV (doesn't change across timesteps)
"""

from typing import Optional, Tuple, Protocol, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import nnx

from .config import DistriFusionConfig, compute_local_sequence_range


# ==============================================================================
# KV Gather Interface (User implements this for JAX SPMD)
# ==============================================================================

class KVGatherInterface(Protocol):
    """Interface for KV gathering operations.

    The user should implement this interface for their specific JAX SPMD setup.
    This allows different implementations for different hardware (TPU, GPU) and
    sharding strategies.
    """

    def gather_kv(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array]:
        """Synchronously gather K and V from all devices.

        Args:
            k_local: Local key tensor [batch, local_seq_len, num_heads, head_dim].
            v_local: Local value tensor [batch, local_seq_len, num_heads, head_dim].
            axis_name: Name of the axis to gather along (e.g., "data", "model").

        Returns:
            (k_full, v_full): Gathered K and V tensors
                [batch, full_seq_len, num_heads, head_dim].
        """
        ...

    def gather_kv_async(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array, Callable[[], None]]:
        """Asynchronously gather K and V from all devices.

        Used for DistriFusion overlap: start async gather, compute attention
        with stale KV, wait for fresh KV to complete.

        Args:
            k_local: Local key tensor [batch, local_seq_len, num_heads, head_dim].
            v_local: Local value tensor [batch, local_seq_len, num_heads, head_dim].
            axis_name: Name of the axis to gather along.

        Returns:
            (k_full, v_full, wait_fn): Gathered tensors and a function to wait
                for completion. The returned tensors may be placeholders until
                wait_fn() is called.
        """
        ...


@dataclass
class DefaultKVGather:
    """Default implementation of KV gathering using JAX collective ops.

    This is a basic implementation. For production TPU/GPU use,
    you should implement optimized versions using pjit/shard_map.
    """

    def gather_kv(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array]:
        """Synchronous all-gather using jax.lax.all_gather."""
        k_full = jax.lax.all_gather(k_local, axis_name=axis_name, axis=1, tiled=True)
        v_full = jax.lax.all_gather(v_local, axis_name=axis_name, axis=1, tiled=True)
        return k_full, v_full

    def gather_kv_async(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array, Callable[[], None]]:
        """Async all-gather (JAX doesn't have true async, this is a placeholder).

        In practice, you would implement this using:
        - Custom NCCL calls for GPU
        - XLA async collective ops for TPU

        For now, this falls back to synchronous gather.
        """
        k_full, v_full = self.gather_kv(k_local, v_local, axis_name)
        return k_full, v_full, lambda: None


# ==============================================================================
# KV Buffer for DistriFusion (Stale KV Reuse)
# ==============================================================================

class KVBuffer(nnx.Variable):
    """Buffer for storing KV from previous timestep.

    This enables DistriFusion's key technique: using stale KV from timestep t-1
    while computing at timestep t, then updating asynchronously.
    """
    pass


@dataclass
class DistriFusionState:
    """State for DistriFusion across timesteps.

    Attributes:
        counter: Current timestep counter.
        kv_buffers: List of (K, V) buffers for each layer.
        cross_kv_cache: Cached text KV (doesn't change across timesteps).
    """
    counter: int = 0
    kv_buffers: Optional[list] = None
    cross_kv_cache: Optional[Tuple[jax.Array, jax.Array]] = None

    def reset(self):
        """Reset state for new generation."""
        self.counter = 0
        self.kv_buffers = None
        self.cross_kv_cache = None


# ==============================================================================
# DistriFusion Self-Attention
# ==============================================================================

class DistriFusionSelfAttention(nnx.Module):
    """DistriFusion wrapper for self-attention.

    Handles sequence splitting and KV gathering for self-attention.
    After warmup steps, uses stale KV from previous timestep for overlap.

    The attention computation itself is delegated to the wrapped attention module.
    """

    def __init__(
        self,
        attention: nnx.Module,
        config: DistriFusionConfig,
        kv_gather: Optional[KVGatherInterface] = None,
        axis_name: str = "data",
        block_idx: int = 0,
    ):
        """Initialize DistriFusion self-attention wrapper.

        Args:
            attention: The wrapped FlaxWanAttention module.
            config: DistriFusion configuration.
            kv_gather: KV gathering implementation. If None, uses default.
            axis_name: Name of the axis for collective operations.
            block_idx: Index of this block (for debugging).
        """
        self.attention = attention
        self.config = config
        self.kv_gather = kv_gather or DefaultKVGather()
        self.axis_name = axis_name
        self.block_idx = block_idx

        # State
        self.counter = nnx.Variable(0)

        # KV buffer for stale KV reuse
        # Will be initialized on first call with correct shape
        self.k_buffer = nnx.Variable(None)
        self.v_buffer = nnx.Variable(None)

    def reset(self):
        """Reset state for new generation."""
        self.counter.value = 0
        self.k_buffer.value = None
        self.v_buffer.value = None

    def __call__(
        self,
        hidden_states: jax.Array,
        rotary_emb: Optional[jax.Array] = None,
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
        # DistriFusion-specific args
        start_idx: int = 0,
        end_idx: int = 0,
        full_seq_len: int = 0,
        device_idx: int = 0,
    ) -> jax.Array:
        """Forward pass with DistriFusion.

        Args:
            hidden_states: Local hidden states [batch, local_seq_len, dim].
            rotary_emb: Local rotary embeddings.
            deterministic: Whether to use dropout.
            rngs: Random number generators.
            start_idx: Start index in full sequence.
            end_idx: End index in full sequence.
            full_seq_len: Total sequence length.
            device_idx: Index of current device.

        Returns:
            Output hidden states [batch, local_seq_len, dim].
        """
        config = self.config
        attn = self.attention

        batch, local_len, dim = hidden_states.shape
        heads = attn.heads
        head_dim = attn.dim_head

        # Compute Q, K, V for local portion
        # Following FlaxWanAttention pattern
        query_proj = attn.query(hidden_states)
        key_proj = attn.key(hidden_states)
        value_proj = attn.value(hidden_states)

        # Apply QK norm if present
        if attn.qk_norm:
            query_proj = attn.norm_q(query_proj)
            key_proj = attn.norm_k(key_proj)

        # Reshape for attention: [batch, local_len, heads, head_dim]
        q_local = query_proj.reshape(batch, local_len, heads, head_dim)
        k_local = key_proj.reshape(batch, local_len, heads, head_dim)
        v_local = value_proj.reshape(batch, local_len, heads, head_dim)

        # Apply rotary embeddings to local Q and K
        if rotary_emb is not None:
            q_local, k_local = attn._apply_rope(q_local, k_local, rotary_emb)

        # DistriFusion: Gather KV from all devices
        use_stale = (
            self.counter.value > config.warmup_steps
            and config.mode != "full_sync"
        )

        if config.mode == "no_sync":
            # Debug mode: no communication
            k_full = k_local
            v_full = v_local
        elif not use_stale:
            # Warmup: synchronous all-gather
            k_full, v_full = self.kv_gather.gather_kv(
                k_local, v_local, self.axis_name
            )
            # Store for next timestep
            self.k_buffer.value = k_full
            self.v_buffer.value = v_full
        else:
            # DistriFusion: use stale KV from previous timestep
            k_full = self.k_buffer.value
            v_full = self.v_buffer.value

            # Start async update for next timestep
            if config.mode == "default":
                k_new, v_new, wait_fn = self.kv_gather.gather_kv_async(
                    k_local, v_local, self.axis_name
                )
                # Update buffer with local portion immediately
                # (async gather will complete in background)
                # For now, we do sync update since JAX doesn't have true async
                wait_fn()
                self.k_buffer.value = k_new
                self.v_buffer.value = v_new

        # Compute attention: local Q attends to full K, V
        # Reshape for attention op: [batch, heads, seq_len, head_dim]
        q_for_attn = jnp.transpose(q_local, (0, 2, 1, 3))
        k_for_attn = jnp.transpose(k_full, (0, 2, 1, 3))
        v_for_attn = jnp.transpose(v_full, (0, 2, 1, 3))

        # Use the attention op
        attn_output = attn.attention_op.apply_attention(
            q_for_attn.reshape(batch, local_len, heads * head_dim),
            k_for_attn.reshape(batch, -1, heads * head_dim),
            v_for_attn.reshape(batch, -1, heads * head_dim),
        )

        # Output projection
        hidden_states = attn.proj_attn(attn_output)
        hidden_states = attn.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)

        self.counter.value += 1

        return hidden_states


# ==============================================================================
# DistriFusion Cross-Attention
# ==============================================================================

class DistriFusionCrossAttention(nnx.Module):
    """DistriFusion wrapper for cross-attention.

    Cross-attention with text encoder doesn't need KV gathering because:
    1. Text KV is the same across all devices
    2. Text KV is the same across all timesteps (can be cached)

    This wrapper handles:
    - Text KV caching (compute once, reuse across timesteps)
    - Local query attending to full text KV
    """

    def __init__(
        self,
        attention: nnx.Module,
        config: DistriFusionConfig,
        block_idx: int = 0,
    ):
        """Initialize DistriFusion cross-attention wrapper.

        Args:
            attention: The wrapped FlaxWanAttention module.
            config: DistriFusion configuration.
            block_idx: Index of this block (for debugging).
        """
        self.attention = attention
        self.config = config
        self.block_idx = block_idx

        # State for KV caching
        self.counter = nnx.Variable(0)
        self.k_cache = nnx.Variable(None)
        self.v_cache = nnx.Variable(None)

    def reset(self):
        """Reset state for new generation."""
        self.counter.value = 0
        self.k_cache.value = None
        self.v_cache.value = None

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        encoder_attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
    ) -> jax.Array:
        """Forward pass with text KV caching.

        Args:
            hidden_states: Local visual hidden states [batch, local_seq_len, dim].
            encoder_hidden_states: Text encoder output [batch, text_len, dim].
            encoder_attention_mask: Optional attention mask for text.
            deterministic: Whether to use dropout.
            rngs: Random number generators.

        Returns:
            Output hidden states [batch, local_seq_len, dim].
        """
        attn = self.attention
        batch, local_len, dim = hidden_states.shape
        heads = attn.heads
        head_dim = attn.dim_head

        # Query from local visual features
        query_proj = attn.query(hidden_states)
        if attn.qk_norm:
            query_proj = attn.norm_q(query_proj)

        # KV from text (cached after first timestep)
        if self.counter.value == 0 or self.k_cache.value is None:
            key_proj = attn.key(encoder_hidden_states)
            value_proj = attn.value(encoder_hidden_states)
            if attn.qk_norm:
                key_proj = attn.norm_k(key_proj)
            self.k_cache.value = key_proj
            self.v_cache.value = value_proj
        else:
            key_proj = self.k_cache.value
            value_proj = self.v_cache.value

        # Compute attention
        attn_output = attn.attention_op.apply_attention(
            query_proj,
            key_proj,
            value_proj,
            attention_mask=encoder_attention_mask,
        )

        # Output projection
        hidden_states = attn.proj_attn(attn_output)
        hidden_states = attn.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)

        self.counter.value += 1

        return hidden_states
