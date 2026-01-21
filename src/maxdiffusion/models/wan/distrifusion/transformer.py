"""
DistriFusion transformer block wrapper for Wan2.1.

This wraps a WanTransformerBlock to handle:
1. Sequence splitting across devices
2. Self-attention with all-gather KV
3. Cross-attention with text KV caching
4. FFN (no communication needed - token-wise)
"""

from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from flax import nnx

from .config import DistriFusionConfig, get_local_rotary_emb
from .attention import (
    DistriFusionSelfAttention,
    DistriFusionCrossAttention,
    KVGatherInterface,
)


class DistriFusionTransformerBlock(nnx.Module):
    """DistriFusion wrapper for WanTransformerBlock.

    This block processes the local sequence portion and handles:
    - Self-attention: all-gather KV, compute attention locally
    - Cross-attention: cache text KV
    - FFN: local computation (no communication)
    """

    def __init__(
        self,
        block: nnx.Module,
        config: DistriFusionConfig,
        kv_gather: Optional[KVGatherInterface] = None,
        axis_name: str = "data",
        block_idx: int = 0,
    ):
        """Initialize DistriFusion transformer block.

        Args:
            block: The wrapped WanTransformerBlock module.
            config: DistriFusion configuration.
            kv_gather: KV gathering implementation.
            axis_name: Name of the axis for collective operations.
            block_idx: Index of this block.
        """
        self.block = block
        self.config = config
        self.block_idx = block_idx

        # Wrap self-attention
        self.df_self_attn = DistriFusionSelfAttention(
            attention=block.attn1,
            config=config,
            kv_gather=kv_gather,
            axis_name=axis_name,
            block_idx=block_idx,
        )

        # Wrap cross-attention
        self.df_cross_attn = DistriFusionCrossAttention(
            attention=block.attn2,
            config=config,
            block_idx=block_idx,
        )

    def reset(self):
        """Reset state for new generation."""
        self.df_self_attn.reset()
        self.df_cross_attn.reset()

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        temb: jax.Array,
        rotary_emb: jax.Array,
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
        encoder_attention_mask: Optional[jax.Array] = None,
        # DistriFusion-specific args
        start_idx: int = 0,
        end_idx: int = 0,
        full_seq_len: int = 0,
        device_idx: int = 0,
    ) -> jax.Array:
        """Forward pass for transformer block with DistriFusion.

        Args:
            hidden_states: Local hidden states [batch, local_seq_len, dim].
            encoder_hidden_states: Text encoder output [batch, text_len, dim].
            temb: Time embedding.
            rotary_emb: Local rotary embeddings for self-attention.
            deterministic: Whether to use dropout.
            rngs: Random number generators.
            encoder_attention_mask: Optional attention mask for text.
            start_idx: Start index in full sequence.
            end_idx: End index in full sequence.
            full_seq_len: Total sequence length.
            device_idx: Index of current device.

        Returns:
            Output hidden states [batch, local_seq_len, dim].
        """
        block = self.block

        # Unpack modulation (same as WanTransformerBlock)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = jnp.split(
            (block.adaln_scale_shift_table.value + temb.astype(jnp.float32)), 6, axis=1
        )

        # 1. Self-attention with DistriFusion
        norm_hidden_states = (
            block.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa
        ).astype(hidden_states.dtype)

        attn_output = self.df_self_attn(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            deterministic=deterministic,
            rngs=rngs,
            start_idx=start_idx,
            end_idx=end_idx,
            full_seq_len=full_seq_len,
            device_idx=device_idx,
        )

        hidden_states = (
            hidden_states.astype(jnp.float32) + attn_output * gate_msa
        ).astype(hidden_states.dtype)

        # 2. Cross-attention with text KV caching
        norm_hidden_states = block.norm2(hidden_states.astype(jnp.float32)).astype(
            hidden_states.dtype
        )

        attn_output = self.df_cross_attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            deterministic=deterministic,
            rngs=rngs,
        )

        hidden_states = hidden_states + attn_output

        # 3. FFN (local computation, no communication needed)
        norm_hidden_states = (
            block.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa
        ).astype(hidden_states.dtype)

        ff_output = block.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)

        hidden_states = (
            hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa
        ).astype(hidden_states.dtype)

        return hidden_states


class DistriFusionWanModel(nnx.Module):
    """DistriFusion wrapper for WanModel.

    Wraps the full WanModel for sequence-parallel inference.
    Each device processes a portion of the sequence.
    """

    def __init__(
        self,
        model: nnx.Module,
        config: DistriFusionConfig,
        kv_gather: Optional[KVGatherInterface] = None,
        axis_name: str = "data",
    ):
        """Initialize DistriFusion model wrapper.

        Args:
            model: The WanModel to wrap.
            config: DistriFusion configuration.
            kv_gather: KV gathering implementation.
            axis_name: Name of the axis for collective operations.
        """
        self.model = model
        self.config = config
        self.axis_name = axis_name

        # Wrap transformer blocks
        if model.scan_layers:
            # For scan layers, we need to handle differently
            # The blocks are vmapped, so we wrap the underlying block
            self.df_blocks = None  # Will use scan-compatible approach
            self._use_scan = True
        else:
            # For non-scan layers, wrap each block
            self.df_blocks = [
                DistriFusionTransformerBlock(
                    block=block,
                    config=config,
                    kv_gather=kv_gather,
                    axis_name=axis_name,
                    block_idx=i,
                )
                for i, block in enumerate(model.blocks)
            ]
            self._use_scan = False

        # Counter for timestep tracking
        self.counter = nnx.Variable(0)

    def reset(self):
        """Reset state for new generation."""
        self.counter.value = 0
        if self.df_blocks is not None:
            for block in self.df_blocks:
                block.reset()

    def __call__(
        self,
        hidden_states: jax.Array,
        timestep: jax.Array,
        encoder_hidden_states: jax.Array,
        encoder_hidden_states_image: Optional[jax.Array] = None,
        return_dict: bool = True,
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
        # DistriFusion-specific args
        device_idx: int = 0,
    ) -> jax.Array:
        """Forward pass with DistriFusion.

        This handles sequence splitting internally based on device_idx.

        Args:
            hidden_states: Full input tensor [batch, channels, frames, height, width].
            timestep: Current timestep.
            encoder_hidden_states: Text encoder output.
            encoder_hidden_states_image: Optional image encoder output.
            return_dict: Whether to return dict (ignored, always returns tensor).
            deterministic: Whether to use dropout.
            rngs: Random number generators.
            device_idx: Index of current device (0 to n_devices-1).

        Returns:
            Output tensor (local portion) [batch, channels, frames, height, width].
        """
        model = self.model
        config = self.config

        # Get dimensions
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = model.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        full_seq_len = post_patch_num_frames * post_patch_height * post_patch_width

        # Compute local sequence range
        local_seq_len = full_seq_len // config.n_devices
        start_idx = device_idx * local_seq_len
        if device_idx == config.n_devices - 1:
            end_idx = full_seq_len
        else:
            end_idx = start_idx + local_seq_len

        # Transpose and patch embedding
        hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
        rotary_emb = model.rope(hidden_states)
        hidden_states = model.patch_embedding(hidden_states)
        hidden_states = jax.lax.collapse(hidden_states, 1, -1)

        # Split to local portion
        hidden_states = hidden_states[:, start_idx:end_idx, :]

        # Get local rotary embeddings
        local_rotary_emb = get_local_rotary_emb(
            rotary_emb,
            start_idx,
            end_idx,
            (post_patch_num_frames, post_patch_height, post_patch_width),
            model.config.attention_head_dim,
        )

        # Condition embeddings
        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
            encoder_attention_mask,
        ) = model.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = jnp.concatenate(
                [encoder_hidden_states_image, encoder_hidden_states], axis=1
            )
            if encoder_attention_mask is not None:
                text_mask = jnp.ones(
                    (
                        encoder_hidden_states.shape[0],
                        encoder_hidden_states.shape[1] - encoder_hidden_states_image.shape[1],
                    ),
                    dtype=jnp.int32,
                )
                encoder_attention_mask = jnp.concatenate(
                    [encoder_attention_mask, text_mask], axis=1
                )
            encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)

        # Process through DistriFusion blocks
        if self._use_scan:
            # For scan layers, we need a different approach
            # This is a placeholder - actual implementation depends on
            # how scan is used with DistriFusion state
            raise NotImplementedError(
                "DistriFusion with scan_layers not yet implemented. "
                "Please use scan_layers=False for now."
            )
        else:
            for df_block in self.df_blocks:
                hidden_states = df_block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    local_rotary_emb,
                    deterministic=deterministic,
                    rngs=rngs,
                    encoder_attention_mask=encoder_attention_mask,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    full_seq_len=full_seq_len,
                    device_idx=device_idx,
                )

        # Final normalization and projection (local)
        shift, scale = jnp.split(
            model.scale_shift_table.value + jnp.expand_dims(temb, axis=1), 2, axis=1
        )
        hidden_states = (
            model.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift
        ).astype(hidden_states.dtype)
        hidden_states = model.proj_out(hidden_states)

        # All-gather to reconstruct full sequence
        # This is where user implements the final gather
        hidden_states = jax.lax.all_gather(
            hidden_states, axis_name=self.axis_name, axis=1, tiled=True
        )

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
        hidden_states = jax.lax.collapse(hidden_states, 6, None)
        hidden_states = jax.lax.collapse(hidden_states, 4, 6)
        hidden_states = jax.lax.collapse(hidden_states, 2, 4)

        self.counter.value += 1

        return hidden_states


def wrap_wan_model_distrifusion(
    model: nnx.Module,
    config: DistriFusionConfig,
    kv_gather: Optional[KVGatherInterface] = None,
    axis_name: str = "data",
) -> DistriFusionWanModel:
    """Wrap a WanModel with DistriFusion for sequence-parallel inference.

    Args:
        model: WanModel instance.
        config: DistriFusion configuration.
        kv_gather: KV gathering implementation (user provides for SPMD).
        axis_name: Name of the axis for collective operations.

    Returns:
        DistriFusion-wrapped model.
    """
    return DistriFusionWanModel(
        model=model,
        config=config,
        kv_gather=kv_gather,
        axis_name=axis_name,
    )
