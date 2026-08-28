import ctypes
from math import prod
from typing import Optional, Sequence, Tuple

import torch


def _ctype_for_dtype(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return ctypes.c_uint16
    if dtype == torch.float32:
        return ctypes.c_float
    if dtype == torch.float64:
        return ctypes.c_double
    if dtype == torch.int8:
        return ctypes.c_int8
    if dtype == torch.uint8:
        return ctypes.c_uint8
    if dtype == torch.int16:
        return ctypes.c_int16
    if dtype == torch.int32:
        return ctypes.c_int32
    if dtype == torch.int64:
        return ctypes.c_int64
    if dtype == torch.bool:
        return ctypes.c_bool
    raise TypeError(f"unsupported shm tensor dtype: {dtype}")


def create_shm_tensor(
    shape: Sequence[int], dtype: torch.dtype, device_id: int = 0, name: str = "",
) -> Tuple[torch.Tensor, int, int]:
    """Create host shared memory and register it to an NPU device.

    Returns ``(host_tensor, host_ptr, dev_ptr)``. ``host_tensor`` is a CPU
    tensor backed by the registered shared memory. ``dev_ptr`` is the
    device-visible address and can be passed to sparse KV kernels through
    ``src_ptr``/``dst_ptr``.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    numel = int(prod(shape_tuple))
    elem_size = torch.empty((), dtype=dtype).element_size()
    size = numel * elem_size
    if size <= 0:
        raise ValueError(f"shm tensor size must be positive, got shape={shape_tuple}")

    host_ptr, dev_ptr = torch.ops.npu.shm_allocator_create_and_register(
        size, device_id, name
    )
    buffer_type = _ctype_for_dtype(dtype) * numel
    buffer = buffer_type.from_address(host_ptr)
    tensor = torch.frombuffer(buffer, dtype=dtype).view(shape_tuple)
    if tensor.element_size() != elem_size:
        raise RuntimeError("shm tensor element size mismatch")
    tensor.zero_()
    return tensor, int(host_ptr), int(dev_ptr)


def free_shm(device_id: int = 0) -> None:
    """Free all shared-memory allocations registered by this process."""
    torch.ops.npu.shm_allocator_free_all(device_id)


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> Tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


def unidex_copy_inplace(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
    dst_ptr: Optional[int] = None,
) -> torch.Tensor:
    """Copy selected logical rows from ``src`` into ``dst`` in place.

    ``src_ptr`` and ``dst_ptr`` may override the Tensor addresses with
    device-visible shared-memory addresses. The caller owns those allocations
    and must keep them alive until work on the current NPU stream completes.
    """
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )

    torch.ops.npu.unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        src_block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst


def unidex_split_copy_inplace(
    src: torch.Tensor,
    dst_nope: torch.Tensor,
    dst_rope: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy indexed combined-KV rows directly into contiguous NoPE/RoPE rows."""

    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, nope_bytes = _infer_rows_and_block_bytes(
        dst_nope, dst_address_ndims, "dst_nope"
    )
    rope_rows, rope_bytes = _infer_rows_and_block_bytes(
        dst_rope, dst_address_ndims, "dst_rope"
    )
    if rope_rows != dst_rows:
        raise ValueError(
            f"dst_nope and dst_rope must have the same row count, got {dst_rows} and {rope_rows}"
        )
    if src_block_bytes != nope_bytes + rope_bytes:
        raise ValueError(
            "src row bytes must equal dst_nope + dst_rope row bytes, got "
            f"{src_block_bytes}, {nope_bytes}, and {rope_bytes}"
        )
    if src.dtype != dst_nope.dtype or src.dtype != dst_rope.dtype:
        raise ValueError("src, dst_nope, and dst_rope must have the same dtype")
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )

    torch.ops.npu.unidex_split_copy(
        src,
        dst_nope,
        dst_rope,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        nope_bytes,
        rope_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
    )
    return dst_nope, dst_rope


def unidex_split_copy_promote_inplace(
    src: torch.Tensor,
    dst_nope: torch.Tensor,
    dst_rope: torch.Tensor,
    hot_cache: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    hot_dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    hot_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy Host misses into SFA buffers and promote the same rows into hot cache."""

    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, nope_bytes = _infer_rows_and_block_bytes(
        dst_nope, dst_address_ndims, "dst_nope"
    )
    rope_rows, rope_bytes = _infer_rows_and_block_bytes(
        dst_rope, dst_address_ndims, "dst_rope"
    )
    hot_rows, hot_block_bytes = _infer_rows_and_block_bytes(
        hot_cache, hot_address_ndims, "hot_cache"
    )
    if rope_rows != dst_rows:
        raise ValueError(
            f"dst_nope and dst_rope must have the same row count, got {dst_rows} and {rope_rows}"
        )
    if src_block_bytes != nope_bytes + rope_bytes or hot_block_bytes != src_block_bytes:
        raise ValueError(
            "src/hot row bytes must equal dst_nope + dst_rope row bytes, got "
            f"src={src_block_bytes}, hot={hot_block_bytes}, "
            f"nope={nope_bytes}, rope={rope_bytes}"
        )
    if not (src.dtype == dst_nope.dtype == dst_rope.dtype == hot_cache.dtype):
        raise ValueError(
            "src, SFA destinations, and hot_cache must have the same dtype"
        )
    if not (
        src_index.numel()
        == dst_index.numel()
        == hot_dst_index.numel()
        == valid_mask.numel()
    ):
        raise ValueError("all copy descriptors must have the same length")

    torch.ops.npu.unidex_split_copy_promote(
        src,
        dst_nope,
        dst_rope,
        hot_cache,
        src_index,
        dst_index,
        hot_dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        hot_rows,
        nope_bytes,
        rope_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
    )
    return dst_nope, dst_rope, hot_cache


def slot_map_lookup(
    slot_map: torch.Tensor,
    req_indices: torch.Tensor,
    topk_indices: torch.Tensor,
    block_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cache-hit flags and slot positions for ``topk_indices``."""
    token_on_device = torch.empty_like(topk_indices, dtype=torch.int32)
    device_token_pos = torch.empty_like(topk_indices, dtype=torch.int32)
    torch.ops.npu.slot_map_lookup(
        slot_map,
        req_indices,
        topk_indices,
        token_on_device,
        device_token_pos,
        block_dim,
    )
    return token_on_device, device_token_pos


def sparse_kv_partition_plan_inplace(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    device_cache_row_indices: torch.Tensor,
    slot_map_row_indices: torch.Tensor,
    valid_topk_mask: torch.Tensor,
    hit_sparse_indices: torch.Tensor,
    miss_sparse_indices: torch.Tensor,
    hit_counts: torch.Tensor,
    miss_counts: torch.Tensor,
    hit_src_indices: torch.Tensor,
    miss_src_indices: torch.Tensor,
    miss_hot_dst_indices: torch.Tensor,
    hit_valid_mask: torch.Tensor,
    miss_valid_mask: torch.Tensor,
    slot_map_flat_indices: torch.Tensor,
    slot_map_slot_values: torch.Tensor,
    max_context_len: int,
    slot_map_width: int,
    block_dim: int = 0,
):
    """Build fixed-shape sparse KV communication descriptors in one NPU op.

    This consumes ``slot_map_lookup`` outputs. It preserves the physical slots
    of hits, assigns misses to free hot-cache slots, creates compact SFA index
    lists, and writes all descriptors needed by hit D2D, miss H2D+promotion,
    and the following slot-map publication. All output tensors are caller-owned
    so their addresses remain stable across NPUGraph replay.
    """
    torch.ops.npu.sparse_kv_partition_plan(
        token_on_device,
        device_token_pos,
        topk_indices,
        device_cache_row_indices,
        slot_map_row_indices,
        valid_topk_mask,
        hit_sparse_indices,
        miss_sparse_indices,
        hit_counts,
        miss_counts,
        hit_src_indices,
        miss_src_indices,
        miss_hot_dst_indices,
        hit_valid_mask,
        miss_valid_mask,
        slot_map_flat_indices,
        slot_map_slot_values,
        max_context_len,
        slot_map_width,
        block_dim,
    )
    return (
        hit_sparse_indices,
        miss_sparse_indices,
        hit_counts,
        miss_counts,
        hit_src_indices,
        miss_src_indices,
        miss_hot_dst_indices,
        hit_valid_mask,
        miss_valid_mask,
        slot_map_flat_indices,
        slot_map_slot_values,
    )


def sparse_kv_partition_plan(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    device_cache_row_indices: torch.Tensor,
    slot_map_row_indices: torch.Tensor,
    valid_topk_mask: torch.Tensor,
    max_context_len: int,
    slot_map_width: int,
    block_dim: int = 0,
):
    """Allocate and return the outputs of :func:`sparse_kv_partition_plan_inplace`."""
    if token_on_device.dim() != 2:
        raise ValueError(
            "token_on_device must have shape [B, K], got "
            f"{tuple(token_on_device.shape)}"
        )
    batch_size, topk = token_on_device.shape
    plan_size = batch_size * topk
    options = dict(device=token_on_device.device)
    outputs = (
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
        torch.empty((batch_size,), dtype=torch.int32, **options),
        torch.empty((batch_size,), dtype=torch.int32, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((plan_size,), dtype=torch.bool, **options),
        torch.empty((plan_size,), dtype=torch.bool, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
    )
    return sparse_kv_partition_plan_inplace(
        token_on_device,
        device_token_pos,
        topk_indices,
        device_cache_row_indices,
        slot_map_row_indices,
        valid_topk_mask,
        *outputs,
        max_context_len,
        slot_map_width,
        block_dim,
    )


def sparse_kv_partition_plan_parallel_inplace(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    device_cache_row_indices: torch.Tensor,
    slot_map_row_indices: torch.Tensor,
    valid_topk_mask: torch.Tensor,
    hit_sparse_indices: torch.Tensor,
    miss_sparse_indices: torch.Tensor,
    hit_counts: torch.Tensor,
    miss_counts: torch.Tensor,
    hit_src_indices: torch.Tensor,
    miss_src_indices: torch.Tensor,
    miss_hot_dst_indices: torch.Tensor,
    hit_valid_mask: torch.Tensor,
    miss_valid_mask: torch.Tensor,
    slot_map_flat_indices: torch.Tensor,
    slot_map_slot_values: torch.Tensor,
    tile_hit_counts: torch.Tensor,
    tile_miss_counts: torch.Tensor,
    tile_hit_offsets: torch.Tensor,
    tile_miss_offsets: torch.Tensor,
    selected_slots: torch.Tensor,
    tile_occupied_bitmaps: torch.Tensor,
    occupied_bitmaps: torch.Tensor,
    free_slot_prefixes: torch.Tensor,
    max_context_len: int,
    slot_map_width: int,
    block_dim: int = 0,
):
    """Run the fixed-address three-kernel parallel partition planner.

    Classification and stable scatter use 64-entry tiles across all AIVs. A
    small request-level scan computes tile offsets and reduces resident-slot
    bitmaps; stable scatter assigns misses to free slots in parallel. The public
    outputs are identical to :func:`sparse_kv_partition_plan_inplace`; the
    additional tensors are graph-stable workspaces owned by the caller.
    """
    torch.ops.npu.sparse_kv_partition_plan_parallel(
        token_on_device,
        device_token_pos,
        topk_indices,
        device_cache_row_indices,
        slot_map_row_indices,
        valid_topk_mask,
        hit_sparse_indices,
        miss_sparse_indices,
        hit_counts,
        miss_counts,
        hit_src_indices,
        miss_src_indices,
        miss_hot_dst_indices,
        hit_valid_mask,
        miss_valid_mask,
        slot_map_flat_indices,
        slot_map_slot_values,
        tile_hit_counts,
        tile_miss_counts,
        tile_hit_offsets,
        tile_miss_offsets,
        selected_slots,
        tile_occupied_bitmaps,
        occupied_bitmaps,
        free_slot_prefixes,
        max_context_len,
        slot_map_width,
        block_dim,
    )
    return (
        hit_sparse_indices,
        miss_sparse_indices,
        hit_counts,
        miss_counts,
        hit_src_indices,
        miss_src_indices,
        miss_hot_dst_indices,
        hit_valid_mask,
        miss_valid_mask,
        slot_map_flat_indices,
        slot_map_slot_values,
    )


def sparse_kv_partition_plan_parallel(
    token_on_device: torch.Tensor,
    device_token_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    device_cache_row_indices: torch.Tensor,
    slot_map_row_indices: torch.Tensor,
    valid_topk_mask: torch.Tensor,
    max_context_len: int,
    slot_map_width: int,
    block_dim: int = 0,
):
    """Allocate outputs/workspaces and run the three-kernel planner."""
    if token_on_device.dim() != 2:
        raise ValueError(
            "token_on_device must have shape [B, K], got "
            f"{tuple(token_on_device.shape)}"
        )
    batch_size, topk = token_on_device.shape
    if topk % 64 != 0:
        raise ValueError(f"topk must be divisible by 64, got {topk}")
    plan_size = batch_size * topk
    tile_shape = (batch_size, topk // 64)
    options = dict(device=token_on_device.device)
    outputs = (
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
        torch.empty((batch_size,), dtype=torch.int32, **options),
        torch.empty((batch_size,), dtype=torch.int32, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((plan_size,), dtype=torch.bool, **options),
        torch.empty((plan_size,), dtype=torch.bool, **options),
        torch.empty((plan_size,), dtype=torch.int64, **options),
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
    )
    workspaces = (
        torch.empty(tile_shape, dtype=torch.int32, **options),
        torch.empty(tile_shape, dtype=torch.int32, **options),
        torch.empty(tile_shape, dtype=torch.int32, **options),
        torch.empty(tile_shape, dtype=torch.int32, **options),
        torch.empty((batch_size, topk), dtype=torch.int32, **options),
        torch.empty(
            (batch_size, topk // 64, topk // 32), dtype=torch.int32, **options,
        ),
        torch.empty((batch_size, topk // 32), dtype=torch.int32, **options),
        torch.empty((batch_size, topk // 32), dtype=torch.int32, **options),
    )
    return sparse_kv_partition_plan_parallel_inplace(
        token_on_device,
        device_token_pos,
        topk_indices,
        device_cache_row_indices,
        slot_map_row_indices,
        valid_topk_mask,
        *outputs,
        *workspaces,
        max_context_len,
        slot_map_width,
        block_dim,
    )


def sfa_state_merge_inplace(
    hit_output: torch.Tensor,
    hit_max: torch.Tensor,
    hit_sum: torch.Tensor,
    miss_output: torch.Tensor,
    miss_max: torch.Tensor,
    miss_sum: torch.Tensor,
    hit_counts: torch.Tensor,
    miss_counts: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Merge hit/miss SFA states into a fixed-address output tensor.

    ``hit_output`` and ``miss_output`` use BSND layout ``[B, S, H, D]``.
    The FP32 max/sum statistics use ``[B, 1, S, H]`` and counts use int32
    ``[B]``. All tensors must use the base NPU ND format, and ``D`` must be a
    multiple of 16. Non-positive counts mark an empty partition. The merge is
    performed in FP32 and cast back to the output dtype.
    """
    return torch.ops.npu.sfa_state_merge(
        hit_output,
        hit_max,
        hit_sum,
        miss_output,
        miss_max,
        miss_sum,
        hit_counts,
        miss_counts,
        output,
    )
