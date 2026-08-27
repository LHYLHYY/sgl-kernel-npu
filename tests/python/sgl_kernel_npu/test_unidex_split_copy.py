import unittest
import uuid

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    free_shm,
    unidex_split_copy_inplace,
    unidex_split_copy_promote_inplace,
)


class TestUnidexSplitCopy(unittest.TestCase):
    def _source(self, rows, dtype, device):
        return (
            torch.arange(rows * 576, dtype=torch.float32)
            .reshape(rows, 576)
            .to(device=device, dtype=dtype)
        )

    def test_device_split_copy(self):
        for dtype in (torch.float16, torch.bfloat16):
            src = self._source(8, dtype, "npu")
            dst_nope = torch.full((6, 512), -1, dtype=dtype, device="npu")
            dst_rope = torch.full((6, 64), -1, dtype=dtype, device="npu")
            src_index = torch.tensor([7, 2, 5, 0], dtype=torch.int64, device="npu")
            dst_index = torch.tensor([0, 4, 2, 5], dtype=torch.int64, device="npu")
            valid = torch.tensor([True, False, True, True], device="npu")

            expected_nope = dst_nope.clone()
            expected_rope = dst_rope.clone()
            expected_nope[0] = src[7, :512]
            expected_rope[0] = src[7, 512:]
            expected_nope[2] = src[5, :512]
            expected_rope[2] = src[5, 512:]
            expected_nope[5] = src[0, :512]
            expected_rope[5] = src[0, 512:]

            outputs = unidex_split_copy_inplace(
                src,
                dst_nope,
                dst_rope,
                src_index,
                dst_index,
                valid,
                src_address_ndims=1,
                dst_address_ndims=1,
                block_dim=8,
            )
            torch.npu.synchronize()
            self.assertIs(outputs[0], dst_nope)
            self.assertIs(outputs[1], dst_rope)
            self.assertTrue(torch.equal(dst_nope, expected_nope))
            self.assertTrue(torch.equal(dst_rope, expected_rope))

    def test_registered_host_split_copy_promotes_to_hot_cache(self):
        device_id = torch.npu.current_device()
        dtype = torch.bfloat16
        source_cpu = self._source(8, dtype, "cpu")
        shm_name = f"unidex_split_copy_{uuid.uuid4().hex}"
        host_src, _, host_dev_ptr = create_shm_tensor(
            source_cpu.shape, dtype, device_id=device_id, name=shm_name
        )
        host_src.copy_(source_cpu)

        dst_nope = torch.full((6, 512), -1, dtype=dtype, device="npu")
        dst_rope = torch.full((6, 64), -1, dtype=dtype, device="npu")
        hot_cache = torch.full((10, 576), -1, dtype=dtype, device="npu")
        src_index = torch.tensor([6, 1, 4, 0], dtype=torch.int64, device="npu")
        dst_index = torch.tensor([3, 0, 5, 2], dtype=torch.int64, device="npu")
        hot_dst_index = torch.tensor([9, 7, 1, 4], dtype=torch.int64, device="npu")
        valid = torch.tensor([True, False, True, True], device="npu")

        expected_nope = dst_nope.clone()
        expected_rope = dst_rope.clone()
        expected_hot = hot_cache.clone()
        for mapping_index in (0, 2, 3):
            source_row = int(src_index[mapping_index].item())
            attention_row = int(dst_index[mapping_index].item())
            hot_row = int(hot_dst_index[mapping_index].item())
            row = source_cpu[source_row].to("npu")
            expected_nope[attention_row] = row[:512]
            expected_rope[attention_row] = row[512:]
            expected_hot[hot_row] = row

        try:
            outputs = unidex_split_copy_promote_inplace(
                host_src,
                dst_nope,
                dst_rope,
                hot_cache,
                src_index,
                dst_index,
                hot_dst_index,
                valid,
                src_address_ndims=1,
                dst_address_ndims=1,
                hot_address_ndims=1,
                block_dim=8,
                src_ptr=host_dev_ptr,
            )
            torch.npu.synchronize()
            self.assertIs(outputs[0], dst_nope)
            self.assertIs(outputs[1], dst_rope)
            self.assertIs(outputs[2], hot_cache)
            self.assertTrue(torch.equal(dst_nope, expected_nope))
            self.assertTrue(torch.equal(dst_rope, expected_rope))
            self.assertTrue(torch.equal(hot_cache, expected_hot))
        finally:
            torch.npu.synchronize()
            free_shm(device_id)

    def test_registered_host_promote_survives_npugraph_dynamic_replay(self):
        device_id = torch.npu.current_device()
        dtype = torch.bfloat16
        shm_name = f"unidex_split_copy_graph_{uuid.uuid4().hex}"
        host_src, _, host_dev_ptr = create_shm_tensor(
            (8, 576), dtype, device_id=device_id, name=shm_name
        )
        dst_nope = torch.empty((6, 512), dtype=dtype, device="npu")
        dst_rope = torch.empty((6, 64), dtype=dtype, device="npu")
        hot_cache = torch.empty((10, 576), dtype=dtype, device="npu")
        src_index = torch.empty(4, dtype=torch.int64, device="npu")
        dst_index = torch.empty(4, dtype=torch.int64, device="npu")
        hot_dst_index = torch.empty(4, dtype=torch.int64, device="npu")
        valid = torch.empty(4, dtype=torch.bool, device="npu")
        graph = None

        def load_case(seed, src_rows, dst_rows, hot_rows, mask):
            torch.npu.synchronize()
            source = self._source(8, dtype, "cpu") + seed
            host_src.copy_(source)
            src_index.copy_(torch.tensor(src_rows, dtype=torch.int64, device="npu"))
            dst_index.copy_(torch.tensor(dst_rows, dtype=torch.int64, device="npu"))
            hot_dst_index.copy_(torch.tensor(hot_rows, dtype=torch.int64, device="npu"))
            valid.copy_(torch.tensor(mask, dtype=torch.bool, device="npu"))
            dst_nope.fill_(-1)
            dst_rope.fill_(-1)
            hot_cache.fill_(-1)
            torch.npu.synchronize()

            expected_nope = torch.full((6, 512), -1, dtype=dtype)
            expected_rope = torch.full((6, 64), -1, dtype=dtype)
            expected_hot = torch.full((10, 576), -1, dtype=dtype)
            for source_row, attention_row, hot_row, is_valid in zip(
                src_rows, dst_rows, hot_rows, mask
            ):
                if is_valid:
                    expected_nope[attention_row] = source[source_row, :512]
                    expected_rope[attention_row] = source[source_row, 512:]
                    expected_hot[hot_row] = source[source_row]
            return (
                expected_nope.to("npu"),
                expected_rope.to("npu"),
                expected_hot.to("npu"),
            )

        def run_once():
            unidex_split_copy_promote_inplace(
                host_src,
                dst_nope,
                dst_rope,
                hot_cache,
                src_index,
                dst_index,
                hot_dst_index,
                valid,
                src_address_ndims=1,
                dst_address_ndims=1,
                hot_address_ndims=1,
                block_dim=8,
                src_ptr=host_dev_ptr,
            )

        try:
            load_case(1, (7, 2, 5, 0), (0, 4, 2, 5), (8, 6, 4, 2), (1, 1, 0, 1))
            capture_stream = torch.npu.Stream()
            capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(capture_stream):
                for _ in range(2):
                    run_once()
            torch.npu.synchronize()

            graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                graph,
                pool=torch.npu.graph_pool_handle(),
                stream=capture_stream,
                auto_dispatch_capture=True,
            ):
                run_once()
            torch.npu.synchronize()

            for case in (
                (11, (6, 1, 4, 0), (3, 0, 5, 2), (9, 7, 1, 4), (1, 0, 1, 1)),
                (23, (0, 3, 7, 2), (5, 1, 0, 4), (2, 8, 6, 0), (1, 1, 1, 0)),
            ):
                expected = load_case(*case)
                graph.replay()
                torch.npu.synchronize()
                self.assertTrue(torch.equal(dst_nope, expected[0]))
                self.assertTrue(torch.equal(dst_rope, expected[1]))
                self.assertTrue(torch.equal(hot_cache, expected[2]))
        finally:
            torch.npu.synchronize()
            if graph is not None and hasattr(graph, "reset"):
                graph.reset()
                torch.npu.synchronize()
            free_shm(device_id)


if __name__ == "__main__":
    unittest.main()
