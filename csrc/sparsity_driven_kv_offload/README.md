# Sparsity-Driven KV Offloading Primitives

This directory contains the Ascend NPU kernel-layer primitives for the
[Sparsity-Driven KV Offloading RFC](https://github.com/sgl-project/sglang/issues/31779).
It does not include the SGLang runtime cache manager or sparse-attention backend
adapter.

## Modules

| Module | Purpose |
| --- | --- |
| `shm_allocator` | Allocates host-backed storage and registers it with the NPU, exposing a stable device-visible address. |
| `unidex_copy` | Performs masked indexed row copies for D2D, H2D, and D2H KV movement. |
| `slot_map_lookup` | Resolves sparse top-k logical KV positions against the device-resident slot map. |
| `sparse_kv_partition_plan` | Fuses hit/miss partitioning, sparse-index packing, free-slot assignment, and copy/publish descriptor generation. |
| `sparse_kv_partition_plan_parallel` | Uses classify, request-scan, and stable-scatter kernels to parallelize 64-entry Top-K tiles while preserving the original planner outputs. |
| `sfa_state_merge` | Fuses two SFA output/max/sum states with empty-partition handling into a fixed-address output. |

The intended data path is:

```text
top-k indices
    │
    ▼
slot_map_lookup
    │
    ▼
sparse_kv_partition_plan
    ├── hit descriptors ── device hot KV ──┐
    └── miss descriptors ─ registered host ─┤
                                            ▼
                                     split copy kernels
                                            │
                                            ▼
                                  shared sparse SFA buffers
```

## Python API

The canonical Python API is:

```python
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    free_shm,
    sparse_kv_partition_plan,
    sfa_state_merge_inplace,
    slot_map_lookup,
    unidex_copy_inplace,
)
```

The registered-memory lifecycle is process-local. The supported deployment
model is multiple processes with one NPU device bound to each process.

## Validation

Run the focused correctness and smoke benchmark suite:

```bash
scripts/sparsity_driven_kv_offload/test_unidex_shm.sh 0
```

Run benchmark sweeps:

```bash
scripts/sparsity_driven_kv_offload/sweep_unidex_copy.sh
scripts/sparsity_driven_kv_offload/sweep_slot_map_lookup.sh
```
