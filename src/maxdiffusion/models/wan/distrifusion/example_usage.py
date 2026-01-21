"""
Example usage of DistriFusion for Wan2.1 in JAX.

This file demonstrates how to:
1. Configure DistriFusion
2. Implement the KVGatherInterface for your SPMD setup
3. Wrap the WanModel for sequence-parallel inference

The key interface for user implementation is KVGatherInterface:
- gather_kv(): Synchronous all-gather of K and V
- gather_kv_async(): Async all-gather for overlap (optional)
"""

from typing import Tuple, Callable, Optional
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from functools import partial

from .config import DistriFusionConfig
from .attention import KVGatherInterface
from .transformer import wrap_wan_model_distrifusion


# ==============================================================================
# Example KV Gather Implementation for TPU/GPU
# ==============================================================================

class TPUKVGather:
    """Example KV gather implementation for TPU with pjit/shard_map.

    This is a template for implementing efficient KV gathering on TPU.
    Users should customize this based on their specific sharding strategy.
    """

    def __init__(self, mesh: Mesh, axis_name: str = "data"):
        self.mesh = mesh
        self.axis_name = axis_name

    def gather_kv(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array]:
        """Synchronous all-gather using jax.lax.all_gather.

        Args:
            k_local: [batch, local_seq_len, num_heads, head_dim]
            v_local: [batch, local_seq_len, num_heads, head_dim]
            axis_name: Axis name for collective operation.

        Returns:
            (k_full, v_full): [batch, full_seq_len, num_heads, head_dim]
        """
        # all_gather along sequence dimension (axis=1)
        # tiled=True means the output is concatenated, not replicated
        k_full = jax.lax.all_gather(k_local, axis_name=axis_name, axis=1, tiled=True)
        v_full = jax.lax.all_gather(v_local, axis_name=axis_name, axis=1, tiled=True)
        return k_full, v_full

    def gather_kv_async(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array, Callable[[], None]]:
        """Async all-gather for DistriFusion overlap.

        NOTE: JAX doesn't have true async collectives in the same way as PyTorch.
        For TPU, you can use:
        1. XLA async collectives via experimental APIs
        2. Custom Pallas kernels
        3. Pipeline the computation differently

        For now, this falls back to synchronous gather.
        A production implementation would use XLA's async collective support.
        """
        k_full, v_full = self.gather_kv(k_local, v_local, axis_name)
        return k_full, v_full, lambda: None


class GPUKVGather:
    """Example KV gather implementation for GPU.

    Uses JAX's all_gather which maps to NCCL under the hood.
    """

    def __init__(self, mesh: Mesh, axis_name: str = "data"):
        self.mesh = mesh
        self.axis_name = axis_name

    def gather_kv(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array]:
        """Synchronous all-gather using NCCL."""
        k_full = jax.lax.all_gather(k_local, axis_name=axis_name, axis=1, tiled=True)
        v_full = jax.lax.all_gather(v_local, axis_name=axis_name, axis=1, tiled=True)
        return k_full, v_full

    def gather_kv_async(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array, Callable[[], None]]:
        """Async all-gather (placeholder).

        For true async on GPU, you would need to:
        1. Use JAX's experimental async dispatch
        2. Or use custom CUDA kernels with NCCL async APIs
        """
        k_full, v_full = self.gather_kv(k_local, v_local, axis_name)
        return k_full, v_full, lambda: None


# ==============================================================================
# Example: Setting up DistriFusion with pjit
# ==============================================================================

def create_distrifusion_model(
    wan_model,
    n_devices: int,
    warmup_steps: int = 2,
    mode: str = "default",
    mesh: Optional[Mesh] = None,
):
    """Create a DistriFusion-wrapped WanModel.

    Args:
        wan_model: The WanModel instance to wrap.
        n_devices: Number of devices to split sequence across.
        warmup_steps: Number of warmup steps (sync all-gather).
        mode: Communication mode ("default", "full_sync", "no_sync").
        mesh: JAX mesh. If None, creates a simple mesh.

    Returns:
        (wrapped_model, mesh): The wrapped model and mesh.
    """
    # Create mesh if not provided
    if mesh is None:
        devices = jax.devices()[:n_devices]
        mesh = Mesh(devices, axis_names=("data",))

    # Create config
    config = DistriFusionConfig(
        n_devices=n_devices,
        warmup_steps=warmup_steps,
        mode=mode,
        patch_size=wan_model.config.patch_size,
        attention_head_dim=wan_model.config.attention_head_dim,
    )

    # Create KV gather based on device type
    if jax.devices()[0].platform == "tpu":
        kv_gather = TPUKVGather(mesh=mesh, axis_name="data")
    else:
        kv_gather = GPUKVGather(mesh=mesh, axis_name="data")

    # Wrap model
    wrapped_model = wrap_wan_model_distrifusion(
        model=wan_model,
        config=config,
        kv_gather=kv_gather,
        axis_name="data",
    )

    return wrapped_model, mesh


# ==============================================================================
# Example: Running inference with DistriFusion
# ==============================================================================

def run_inference_with_distrifusion(
    wrapped_model,
    mesh: Mesh,
    hidden_states: jax.Array,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
):
    """Run inference using DistriFusion.

    This example shows how to use shard_map for sequence-parallel inference.
    """

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            PartitionSpec("data", None, None, None, None),  # hidden_states
            PartitionSpec(),  # timestep
            PartitionSpec(None, None, None),  # encoder_hidden_states
        ),
        out_specs=PartitionSpec("data", None, None, None, None),
        check_rep=False,
    )
    def sharded_forward(hidden_states, timestep, encoder_hidden_states):
        # Get device index within the mesh
        device_idx = jax.lax.axis_index("data")

        # Run forward pass
        output = wrapped_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            device_idx=device_idx,
        )

        return output

    return sharded_forward(hidden_states, timestep, encoder_hidden_states)


# ==============================================================================
# Documentation: Interface for User Implementation
# ==============================================================================

"""
## KVGatherInterface

The key interface users need to implement for their JAX SPMD setup:

```python
class KVGatherInterface(Protocol):
    def gather_kv(
        self,
        k_local: jax.Array,  # [batch, local_seq_len, num_heads, head_dim]
        v_local: jax.Array,  # [batch, local_seq_len, num_heads, head_dim]
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array]:
        '''Synchronously gather K and V from all devices.

        Returns (k_full, v_full) with shape [batch, full_seq_len, num_heads, head_dim].
        '''
        ...

    def gather_kv_async(
        self,
        k_local: jax.Array,
        v_local: jax.Array,
        axis_name: str,
    ) -> Tuple[jax.Array, jax.Array, Callable[[], None]]:
        '''Asynchronously gather K and V for DistriFusion overlap.

        Returns (k_full, v_full, wait_fn).
        - k_full, v_full may be placeholders until wait_fn() is called
        - wait_fn() blocks until the async operation completes

        For DistriFusion:
        - Compute attention with stale KV from previous timestep
        - In parallel, start async gather for fresh KV
        - Wait for fresh KV before next timestep
        '''
        ...
```

## Integration with pjit/shard_map

For production use with JAX SPMD:

1. Create a Mesh with your devices
2. Use shard_map to run the sharded forward pass
3. The sequence dimension is sharded across the mesh

```python
mesh = Mesh(devices, axis_names=("data",))

@shard_map(
    mesh=mesh,
    in_specs=(PartitionSpec("data", None, None), ...),
    out_specs=PartitionSpec("data", None, None),
)
def forward(hidden_states, ...):
    device_idx = jax.lax.axis_index("data")
    return wrapped_model(hidden_states, ..., device_idx=device_idx)
```

## XLA Async Collectives (Advanced)

For true async overlap on TPU, you can explore:
1. `jax.experimental.host_callback` for async dispatch
2. Custom Pallas kernels with async communication
3. XLA's internal async collective support

The current implementation provides synchronous gathering, which is correct
but doesn't achieve the full overlap benefit of DistriFusion.
"""
