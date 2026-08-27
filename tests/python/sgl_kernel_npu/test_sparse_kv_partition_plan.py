import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    sparse_kv_partition_plan,
    sparse_kv_partition_plan_inplace,
)


TOPK = 2048
CONTEXT = 4096
SLOT_MAP_WIDTH = 4104


def reference_plan(
    token_on_device,
    device_token_pos,
    topk_indices,
    device_cache_rows,
    slot_map_rows,
    valid_topk,
):
    token_on_device = token_on_device.cpu()
    device_token_pos = device_token_pos.cpu()
    topk_indices = topk_indices.cpu()
    device_cache_rows = device_cache_rows.cpu()
    slot_map_rows = slot_map_rows.cpu()
    valid_topk = valid_topk.cpu()
    batch_size = topk_indices.size(0)

    hit_sparse = torch.full((batch_size, TOPK), -1, dtype=torch.int32)
    miss_sparse = torch.full_like(hit_sparse, -1)
    hit_counts = torch.zeros(batch_size, dtype=torch.int32)
    miss_counts = torch.zeros_like(hit_counts)
    hit_src = torch.zeros(batch_size * TOPK, dtype=torch.int64)
    miss_src = torch.zeros_like(hit_src)
    miss_hot_dst = torch.zeros_like(hit_src)
    hit_valid = torch.zeros(batch_size * TOPK, dtype=torch.bool)
    miss_valid = torch.zeros_like(hit_valid)
    slot_map_flat = torch.zeros_like(hit_src)
    slot_values = torch.full((batch_size, TOPK), -1, dtype=torch.int32)

    for batch in range(batch_size):
        cache_base = int(device_cache_rows[batch]) * TOPK
        host_base = int(device_cache_rows[batch]) * CONTEXT
        slot_map_base = int(slot_map_rows[batch]) * SLOT_MAP_WIDTH
        occupied = set()
        misses = []
        hits = []
        for position in range(TOPK):
            flat = batch * TOPK + position
            logical_token = int(topk_indices[batch, position])
            slot = int(device_token_pos[batch, position])
            valid = bool(valid_topk[batch, position])
            is_hit = (
                valid
                and int(token_on_device[batch, position]) != 0
                and 0 <= slot < TOPK
            )
            is_miss = valid and not is_hit
            hit_valid[flat] = is_hit
            miss_valid[flat] = is_miss
            hit_src[flat] = cache_base + (slot if is_hit else 0)
            miss_src[flat] = host_base + min(max(logical_token, 0), CONTEXT - 1)
            miss_hot_dst[flat] = cache_base
            slot_map_flat[flat] = slot_map_base + (logical_token if valid else CONTEXT)
            if is_hit:
                hits.append(position)
                occupied.add(slot)
                slot_values[batch, position] = slot
            elif is_miss:
                misses.append(position)

        hit_counts[batch] = len(hits)
        miss_counts[batch] = len(misses)
        if hits:
            hit_sparse[batch, : len(hits)] = torch.tensor(hits, dtype=torch.int32)
        else:
            hit_sparse[batch, 0] = 0
        if misses:
            miss_sparse[batch, : len(misses)] = torch.tensor(misses, dtype=torch.int32)
        else:
            miss_sparse[batch, 0] = 0

        free_slots = [slot for slot in range(TOPK) if slot not in occupied]
        for miss_rank, position in enumerate(misses):
            flat = batch * TOPK + position
            assigned = free_slots[miss_rank]
            miss_hot_dst[flat] = cache_base + assigned
            slot_values[batch, position] = assigned

    return (
        hit_sparse,
        miss_sparse,
        hit_counts,
        miss_counts,
        hit_src,
        miss_src,
        miss_hot_dst,
        hit_valid,
        miss_valid,
        slot_map_flat,
        slot_values,
    )


class TestSparseKvPartitionPlan(unittest.TestCase):
    def _make_inputs(self):
        batch_size = 4
        topk = torch.full((batch_size, TOPK), -1, dtype=torch.int32)
        valid = torch.zeros((batch_size, TOPK), dtype=torch.bool)
        token_on_device = torch.zeros((batch_size, TOPK), dtype=torch.int32)
        device_pos = torch.full((batch_size, TOPK), -1, dtype=torch.int32)

        topk[0, :8] = torch.tensor([91, 7, 33, 102, 5, 80, 77, 12])
        valid[0, :8] = True
        for position, slot in ((0, 17), (2, 3), (5, 1500), (7, 9)):
            token_on_device[0, position] = 1
            device_pos[0, position] = slot

        topk[1, :6] = torch.tensor([41, 19, 8, 99, 201, 6])
        valid[1, :6] = True
        token_on_device[1, :6] = 1
        device_pos[1, :6] = torch.tensor([90, 4, 301, 17, 1, 1024])

        topk[3, :6] = torch.tensor([300, 9, 2047, 13, 88, 4095])
        valid[3, :6] = True

        return tuple(
            tensor.to("npu")
            for tensor in (
                token_on_device,
                device_pos,
                topk,
                torch.tensor([2, 1, 0, 3], dtype=torch.int64),
                torch.tensor([2, 1, 4, 3], dtype=torch.int64),
                valid,
            )
        )

    def _empty_outputs(self, batch_size, device="npu"):
        plan_size = batch_size * TOPK
        return (
            torch.empty((batch_size, TOPK), dtype=torch.int32, device=device),
            torch.empty((batch_size, TOPK), dtype=torch.int32, device=device),
            torch.empty((batch_size,), dtype=torch.int32, device=device),
            torch.empty((batch_size,), dtype=torch.int32, device=device),
            torch.empty((plan_size,), dtype=torch.int64, device=device),
            torch.empty((plan_size,), dtype=torch.int64, device=device),
            torch.empty((plan_size,), dtype=torch.int64, device=device),
            torch.empty((plan_size,), dtype=torch.bool, device=device),
            torch.empty((plan_size,), dtype=torch.bool, device=device),
            torch.empty((plan_size,), dtype=torch.int64, device=device),
            torch.empty((batch_size, TOPK), dtype=torch.int32, device=device),
        )

    def test_matches_reference_and_preserves_hit_slots(self):
        inputs = self._make_inputs()
        expected = reference_plan(*inputs)
        for block_dim in (0, 1, 2, 8):
            with self.subTest(block_dim=block_dim):
                actual = sparse_kv_partition_plan(
                    *inputs,
                    max_context_len=CONTEXT,
                    slot_map_width=SLOT_MAP_WIDTH,
                    block_dim=block_dim,
                )
                torch.npu.synchronize()
                for actual_tensor, expected_tensor in zip(actual, expected):
                    self.assertTrue(
                        torch.equal(actual_tensor.cpu(), expected_tensor),
                        msg=(
                            f"mismatch for dtype={actual_tensor.dtype}, "
                            f"shape={tuple(actual_tensor.shape)}, block_dim={block_dim}"
                        ),
                    )

    def test_rejects_non_fixed_topk(self):
        batch = 1
        width = 64
        args = (
            torch.zeros((batch, width), dtype=torch.int32, device="npu"),
            torch.full((batch, width), -1, dtype=torch.int32, device="npu"),
            torch.zeros((batch, width), dtype=torch.int32, device="npu"),
            torch.ones((batch,), dtype=torch.int64, device="npu"),
            torch.ones((batch,), dtype=torch.int64, device="npu"),
            torch.ones((batch, width), dtype=torch.bool, device="npu"),
        )
        with self.assertRaisesRegex(RuntimeError, "requires K=2048"):
            sparse_kv_partition_plan(
                *args, max_context_len=CONTEXT, slot_map_width=SLOT_MAP_WIDTH,
            )

    def test_npugraph_replay_reads_dynamic_plan_inputs(self):
        case_a = self._make_inputs()
        case_b = tuple(tensor.clone() for tensor in case_a)
        case_b[0][0].zero_()
        case_b[1][0].fill_(-1)
        case_b[2][0].fill_(-1)
        case_b[5][0].zero_()
        case_b[2][0, :12] = torch.arange(100, 112, dtype=torch.int32, device="npu")
        case_b[5][0, :12] = True
        case_b[0][0, 1] = 1
        case_b[1][0, 1] = 200
        case_b[0][0, 3] = 1
        case_b[1][0, 3] = 7
        case_b[5][2].zero_()

        static_inputs = tuple(torch.empty_like(tensor) for tensor in case_a)
        static_outputs = self._empty_outputs(case_a[0].size(0))
        input_ptrs = tuple(tensor.data_ptr() for tensor in static_inputs)
        output_ptrs = tuple(tensor.data_ptr() for tensor in static_outputs)

        def load_case(case):
            for static, value in zip(static_inputs, case):
                static.copy_(value)

        def run_once():
            sparse_kv_partition_plan_inplace(
                *static_inputs, *static_outputs, CONTEXT, SLOT_MAP_WIDTH,
            )

        load_case(case_a)
        capture_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            run_once()
            run_once()
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(
            graph, pool=pool, stream=capture_stream, auto_dispatch_capture=True,
        ):
            run_once()
        torch.npu.synchronize()

        try:
            for case in (case_b, case_a):
                expected = reference_plan(*case)
                load_case(case)
                for output in static_outputs:
                    output.zero_()
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                self.assertEqual(
                    tuple(tensor.data_ptr() for tensor in static_inputs), input_ptrs
                )
                self.assertEqual(
                    tuple(tensor.data_ptr() for tensor in static_outputs), output_ptrs
                )
                for actual_tensor, expected_tensor in zip(static_outputs, expected):
                    self.assertTrue(
                        torch.equal(actual_tensor.cpu(), expected_tensor),
                        msg=f"NPUGraph mismatch for shape={tuple(actual_tensor.shape)}",
                    )
        finally:
            torch.npu.synchronize()
            if hasattr(graph, "reset"):
                graph.reset()


if __name__ == "__main__":
    unittest.main()
