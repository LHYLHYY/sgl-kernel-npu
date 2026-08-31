import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    sparse_kv_partition_plan,
    sparse_kv_partition_plan_inplace,
    sparse_kv_partition_plan_parallel,
    sparse_kv_partition_plan_parallel_inplace,
)


TOPK = 2048
CONTEXT = 4096
SLOT_MAP_WIDTH = 4104
PLAN_OUTPUT_NAMES = (
    "hit_sparse_indices",
    "miss_sparse_indices",
    "hit_counts",
    "miss_counts",
    "hit_src_indices",
    "miss_src_indices",
    "miss_hot_dst_indices",
    "hit_valid_mask",
    "miss_valid_mask",
    "slot_map_flat_indices",
    "slot_map_slot_values",
)


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


def reference_plan_with_physical_sparse_indices(*inputs):
    outputs = list(reference_plan(*inputs))
    hit_positions, miss_positions = outputs[:2]
    hit_counts, miss_counts = outputs[2:4]
    slot_values = outputs[-1]
    hit_slots = torch.full_like(hit_positions, -1)
    miss_slots = torch.full_like(miss_positions, -1)

    for batch in range(hit_positions.size(0)):
        hit_count = int(hit_counts[batch])
        miss_count = int(miss_counts[batch])
        if hit_count > 0:
            positions = hit_positions[batch, :hit_count].to(torch.long)
            hit_slots[batch, :hit_count] = slot_values[batch].index_select(
                0, positions
            )
        else:
            hit_slots[batch, 0] = 0
        if miss_count > 0:
            positions = miss_positions[batch, :miss_count].to(torch.long)
            miss_slots[batch, :miss_count] = slot_values[batch].index_select(
                0, positions
            )
        else:
            miss_slots[batch, 0] = 0

    outputs[0] = hit_slots
    outputs[1] = miss_slots
    return tuple(outputs)


class TestSparseKvPartitionPlan(unittest.TestCase):
    def _assert_plan_equal(self, actual, expected, context):
        for name, actual_tensor, expected_tensor in zip(
            PLAN_OUTPUT_NAMES, actual, expected
        ):
            actual_cpu = actual_tensor.cpu()
            if torch.equal(actual_cpu, expected_tensor):
                continue
            mismatch = actual_cpu.reshape(-1) != expected_tensor.reshape(-1)
            first_index = int(torch.nonzero(mismatch, as_tuple=False)[0])
            self.fail(
                f"{context}: output={name}, dtype={actual_tensor.dtype}, "
                f"shape={tuple(actual_tensor.shape)}, first_flat_index={first_index}, "
                f"actual={actual_cpu.reshape(-1)[first_index].item()}, "
                f"expected={expected_tensor.reshape(-1)[first_index].item()}"
            )

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

    def _empty_parallel_workspaces(self, batch_size, device="npu"):
        tile_shape = (batch_size, TOPK // 64)
        bitmap_words = TOPK // 32
        return (
            torch.empty(tile_shape, dtype=torch.int32, device=device),
            torch.empty(tile_shape, dtype=torch.int32, device=device),
            torch.empty(tile_shape, dtype=torch.int32, device=device),
            torch.empty(tile_shape, dtype=torch.int32, device=device),
            torch.empty((batch_size, TOPK), dtype=torch.int32, device=device),
            torch.empty(
                (batch_size, TOPK // 64, bitmap_words),
                dtype=torch.int32,
                device=device,
            ),
            torch.empty((batch_size, bitmap_words), dtype=torch.int32, device=device),
            torch.empty((batch_size, bitmap_words), dtype=torch.int32, device=device),
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
                self._assert_plan_equal(
                    actual, expected, f"single planner block_dim={block_dim}"
                )

    def test_parallel_three_kernel_plan_matches_reference(self):
        inputs = self._make_inputs()
        expected = reference_plan(*inputs)
        for block_dim in (0, 1, 2, 8):
            with self.subTest(block_dim=block_dim):
                actual = sparse_kv_partition_plan_parallel(
                    *inputs,
                    max_context_len=CONTEXT,
                    slot_map_width=SLOT_MAP_WIDTH,
                    block_dim=block_dim,
                )
                torch.npu.synchronize()
                self._assert_plan_equal(
                    actual, expected, f"parallel planner block_dim={block_dim}"
                )

    def test_parallel_plan_can_emit_compact_physical_slots(self):
        inputs = self._make_inputs()
        expected = reference_plan_with_physical_sparse_indices(*inputs)
        actual = sparse_kv_partition_plan_parallel(
            *inputs,
            max_context_len=CONTEXT,
            slot_map_width=SLOT_MAP_WIDTH,
            output_physical_slots=True,
        )
        torch.npu.synchronize()
        self._assert_plan_equal(actual, expected, "parallel physical-slot planner")

    def test_parallel_plan_compacts_across_all_32_tiles(self):
        batch_size = 2
        positions = torch.arange(TOPK, dtype=torch.int32)
        topk = positions.unsqueeze(0).expand(batch_size, TOPK).clone()
        valid = torch.ones((batch_size, TOPK), dtype=torch.bool)
        valid[1, 1537:] = False
        token_on_device = torch.zeros((batch_size, TOPK), dtype=torch.int32)
        device_pos = torch.full((batch_size, TOPK), -1, dtype=torch.int32)

        hit0 = positions.remainder(3) != 1
        hit1 = (positions.remainder(5) == 0) & valid[1]
        token_on_device[0, hit0] = 1
        token_on_device[1, hit1] = 1
        permuted_slots = positions.mul(37).remainder(TOPK)
        device_pos[0, hit0] = permuted_slots[hit0]
        device_pos[1, hit1] = permuted_slots.flip(0)[hit1]

        inputs = tuple(
            tensor.to("npu")
            for tensor in (
                token_on_device,
                device_pos,
                topk,
                torch.tensor([3, 1], dtype=torch.int64),
                torch.tensor([3, 1], dtype=torch.int64),
                valid,
            )
        )
        expected = reference_plan(*inputs)
        actual = sparse_kv_partition_plan_parallel(
            *inputs, max_context_len=CONTEXT, slot_map_width=SLOT_MAP_WIDTH,
        )
        torch.npu.synchronize()
        self._assert_plan_equal(actual, expected, "parallel cross-tile planner")

    def test_parallel_workspace_bitmaps_and_tri_state(self):
        inputs = self._make_inputs()
        batch_size = inputs[0].size(0)
        outputs = self._empty_outputs(batch_size)
        workspaces = self._empty_parallel_workspaces(batch_size)

        sparse_kv_partition_plan_parallel_inplace(
            *inputs, *outputs, *workspaces, CONTEXT, SLOT_MAP_WIDTH,
        )
        torch.npu.synchronize()

        token_on_device, device_pos, _, _, _, valid = (
            tensor.cpu() for tensor in inputs
        )
        hit = valid & (token_on_device != 0) & (device_pos >= 0) & (device_pos < TOPK)
        miss = valid & ~hit
        expected_selected = torch.full((batch_size, TOPK), -1, dtype=torch.int32)
        expected_selected[miss] = -2
        expected_selected[hit] = device_pos[hit]
        torch.testing.assert_close(workspaces[4].cpu(), expected_selected)

        bitmap_words = TOPK // 32
        expected_tile_bitmaps = torch.zeros(
            (batch_size, TOPK // 64, bitmap_words), dtype=torch.int64
        )
        for batch in range(batch_size):
            for position in torch.nonzero(hit[batch], as_tuple=False).flatten():
                position = int(position)
                slot = int(device_pos[batch, position])
                expected_tile_bitmaps[batch, position // 64, slot // 32] |= 1 << (
                    slot % 32
                )
        expected_tile_bitmaps = expected_tile_bitmaps.to(torch.int32)
        torch.testing.assert_close(workspaces[5].cpu(), expected_tile_bitmaps)

        expected_occupied = torch.zeros((batch_size, bitmap_words), dtype=torch.int32)
        for tile in range(TOPK // 64):
            expected_occupied.bitwise_or_(expected_tile_bitmaps[:, tile])
        torch.testing.assert_close(workspaces[6].cpu(), expected_occupied)

        expected_prefix = torch.zeros_like(expected_occupied)
        for batch in range(batch_size):
            free_count = 0
            for word in range(bitmap_words):
                expected_prefix[batch, word] = free_count
                occupied_word = int(expected_occupied[batch, word]) & 0xFFFFFFFF
                free_count += 32 - occupied_word.bit_count()
        torch.testing.assert_close(workspaces[7].cpu(), expected_prefix)

    def test_parallel_full_hit_and_full_miss_rows(self):
        positions = torch.arange(TOPK, dtype=torch.int32)
        topk = positions.unsqueeze(0).expand(2, TOPK).clone()
        valid = torch.ones((2, TOPK), dtype=torch.bool)
        token_on_device = torch.zeros((2, TOPK), dtype=torch.int32)
        token_on_device[0].fill_(1)
        device_pos = torch.full((2, TOPK), -1, dtype=torch.int32)
        device_pos[0] = positions.mul(37).remainder(TOPK)
        inputs = tuple(
            tensor.to("npu")
            for tensor in (
                token_on_device,
                device_pos,
                topk,
                torch.tensor([3, 1], dtype=torch.int64),
                torch.tensor([3, 1], dtype=torch.int64),
                valid,
            )
        )

        expected = reference_plan(*inputs)
        actual = sparse_kv_partition_plan_parallel(
            *inputs, max_context_len=CONTEXT, slot_map_width=SLOT_MAP_WIDTH,
        )
        torch.npu.synchronize()
        self._assert_plan_equal(actual, expected, "parallel full hit/miss rows")

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
                self._assert_plan_equal(
                    static_outputs, expected, "single planner NPUGraph replay"
                )
        finally:
            torch.npu.synchronize()
            if hasattr(graph, "reset"):
                graph.reset()

    def test_parallel_plan_npugraph_replay_reads_dynamic_inputs(self):
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
        static_workspaces = self._empty_parallel_workspaces(case_a[0].size(0))
        input_ptrs = tuple(tensor.data_ptr() for tensor in static_inputs)
        output_ptrs = tuple(tensor.data_ptr() for tensor in static_outputs)
        workspace_ptrs = tuple(tensor.data_ptr() for tensor in static_workspaces)

        def load_case(case):
            for static, value in zip(static_inputs, case):
                static.copy_(value)

        def run_once():
            sparse_kv_partition_plan_parallel_inplace(
                *static_inputs,
                *static_outputs,
                *static_workspaces,
                CONTEXT,
                SLOT_MAP_WIDTH,
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
                self.assertEqual(
                    tuple(tensor.data_ptr() for tensor in static_workspaces),
                    workspace_ptrs,
                )
                self._assert_plan_equal(
                    static_outputs, expected, "parallel planner NPUGraph replay"
                )
        finally:
            torch.npu.synchronize()
            if hasattr(graph, "reset"):
                graph.reset()


if __name__ == "__main__":
    unittest.main()
