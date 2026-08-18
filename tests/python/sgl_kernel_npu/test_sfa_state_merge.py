import unittest

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import sfa_state_merge_inplace

NPU_FORMAT_ND = 2


def _reference_merge(
    hit_output,
    hit_max,
    hit_sum,
    miss_output,
    miss_max,
    miss_sum,
    hit_counts,
    miss_counts,
):
    batch = hit_output.shape[0]
    stats_mask_shape = (batch, 1, 1, 1)
    hit_nonempty = hit_counts.view(stats_mask_shape) > 0
    miss_nonempty = miss_counts.view(stats_mask_shape) > 0
    any_nonempty = hit_nonempty | miss_nonempty

    neg_inf = torch.full_like(hit_max, float("-inf"), dtype=torch.float32)
    hit_max_valid = torch.where(hit_nonempty, hit_max.float(), neg_inf)
    miss_max_valid = torch.where(miss_nonempty, miss_max.float(), neg_inf)
    global_max = torch.maximum(hit_max_valid, miss_max_valid)
    safe_global_max = torch.where(any_nonempty, global_max, 0.0)

    hit_delta = torch.where(hit_nonempty, hit_max.float() - safe_global_max, 0.0)
    miss_delta = torch.where(miss_nonempty, miss_max.float() - safe_global_max, 0.0)
    hit_mass = torch.where(hit_nonempty, hit_sum.float() * torch.exp(hit_delta), 0.0)
    miss_mass = torch.where(
        miss_nonempty, miss_sum.float() * torch.exp(miss_delta), 0.0
    )
    denominator = (hit_mass + miss_mass).clamp_min(torch.finfo(torch.float32).tiny)
    hit_weight = (hit_mass / denominator).permute(0, 2, 3, 1)
    miss_weight = (miss_mass / denominator).permute(0, 2, 3, 1)

    output_mask = any_nonempty.permute(0, 2, 3, 1)
    hit_output_mask = hit_nonempty.permute(0, 2, 3, 1)
    miss_output_mask = miss_nonempty.permute(0, 2, 3, 1)
    hit_values = torch.where(hit_output_mask, hit_output.float(), 0.0)
    miss_values = torch.where(miss_output_mask, miss_output.float(), 0.0)
    merged = hit_values * hit_weight + miss_values * miss_weight
    return torch.where(output_mask, merged, 0.0).to(hit_output.dtype)


class TestSfaStateMerge(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        self.device = torch.device("npu:0")

    def _to_npu_base_format(self, tensor):
        tensor = tensor.to(self.device)
        if tensor.dim() == 4:
            tensor = torch_npu.npu_format_cast(tensor, NPU_FORMAT_ND)
        return tensor

    def _make_case(self, dtype, seed):
        generator = torch.Generator().manual_seed(seed)
        shape = (4, 2, 16, 512)
        stats_shape = (4, 1, 2, 16)
        hit_output = torch.randn(shape, dtype=torch.float32, generator=generator).to(
            dtype
        )
        miss_output = torch.randn(shape, dtype=torch.float32, generator=generator).to(
            dtype
        )
        hit_max = torch.randn(stats_shape, dtype=torch.float32, generator=generator)
        miss_max = torch.randn(stats_shape, dtype=torch.float32, generator=generator)
        # Include large max gaps to exercise stable underflow on the smaller side.
        hit_max[0, :, :, 0] = 100.0
        miss_max[0, :, :, 0] = -100.0
        hit_sum = (
            torch.rand(stats_shape, dtype=torch.float32, generator=generator) + 0.1
        )
        miss_sum = (
            torch.rand(stats_shape, dtype=torch.float32, generator=generator) + 0.1
        )
        hit_counts = torch.tensor([17, 11, 0, 0], dtype=torch.int32)
        miss_counts = torch.tensor([13, 0, 19, 0], dtype=torch.int32)
        return (
            hit_output,
            hit_max,
            hit_sum,
            miss_output,
            miss_max,
            miss_sum,
            hit_counts,
            miss_counts,
        )

    def _run_eager_case(self, dtype):
        cpu_case = self._make_case(dtype, seed=17)
        expected = _reference_merge(*cpu_case)
        npu_case = tuple(self._to_npu_base_format(tensor) for tensor in cpu_case)
        output = torch.full_like(npu_case[0], float("nan"))
        output_address = output.data_ptr()

        result = sfa_state_merge_inplace(*npu_case, output)
        torch.npu.synchronize()

        self.assertEqual(result.data_ptr(), output_address)
        atol = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), rtol=atol, atol=atol
        )
        self.assertTrue(torch.equal(output[1].cpu(), cpu_case[0][1]))
        self.assertTrue(torch.equal(output[2].cpu(), cpu_case[3][2]))
        self.assertTrue(torch.count_nonzero(output[3]).item() == 0)

    def test_eager_fp16_and_bf16(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_eager_case(dtype)

    def test_empty_partition_does_not_read_dummy_output(self):
        cpu_case = list(self._make_case(torch.bfloat16, seed=23))
        cpu_case[0][2].fill_(float("nan"))
        cpu_case[3][1].fill_(float("nan"))
        cpu_case[0][3].fill_(float("nan"))
        cpu_case[3][3].fill_(float("nan"))
        npu_case = tuple(self._to_npu_base_format(tensor) for tensor in cpu_case)
        output = torch.empty_like(npu_case[0])

        sfa_state_merge_inplace(*npu_case, output)
        torch.npu.synchronize()

        self.assertTrue(torch.equal(output[1].cpu(), cpu_case[0][1]))
        self.assertTrue(torch.equal(output[2].cpu(), cpu_case[3][2]))
        self.assertTrue(torch.count_nonzero(output[3]).item() == 0)

    def test_rejects_invalid_contract(self):
        valid = [
            self._to_npu_base_format(tensor)
            for tensor in self._make_case(torch.bfloat16, 29)
        ]
        output = torch.empty_like(valid[0])

        invalid_stats = list(valid)
        invalid_stats[1] = torch_npu.npu_format_cast(
            torch.empty((4, 1, 1, 16), dtype=torch.float32, device=self.device),
            NPU_FORMAT_ND,
        )
        with self.assertRaisesRegex(RuntimeError, "hit_max must have shape"):
            sfa_state_merge_inplace(*invalid_stats, output)

        invalid_stats_dtype = list(valid)
        invalid_stats_dtype[2] = torch_npu.npu_format_cast(
            invalid_stats_dtype[2].to(torch.bfloat16), NPU_FORMAT_ND
        )
        with self.assertRaisesRegex(RuntimeError, "hit_sum must be float32"):
            sfa_state_merge_inplace(*invalid_stats_dtype, output)

        invalid_counts = list(valid)
        invalid_counts[6] = invalid_counts[6].to(torch.int64)
        with self.assertRaisesRegex(RuntimeError, "hit_counts must be int32"):
            sfa_state_merge_inplace(*invalid_counts, output)

        invalid_output = torch_npu.npu_format_cast(
            torch.empty((4, 2, 16, 256), dtype=torch.bfloat16, device=self.device),
            NPU_FORMAT_ND,
        )
        with self.assertRaisesRegex(RuntimeError, "output shape must match"):
            sfa_state_merge_inplace(*valid, invalid_output)

        invalid_head_dim = list(valid)
        invalid_head_dim[0] = torch_npu.npu_format_cast(
            invalid_head_dim[0][..., :511].contiguous(), NPU_FORMAT_ND
        )
        invalid_head_dim[3] = torch_npu.npu_format_cast(
            invalid_head_dim[3][..., :511].contiguous(), NPU_FORMAT_ND
        )
        invalid_head_dim_output = torch.empty_like(invalid_head_dim[0])
        with self.assertRaisesRegex(RuntimeError, "D must be a multiple of 16"):
            sfa_state_merge_inplace(*invalid_head_dim, invalid_head_dim_output)

    def test_npugraph_replay_updates_fixed_output(self):
        dtype = torch.bfloat16
        first_cpu = tuple(tensor.clone() for tensor in self._make_case(dtype, seed=31))
        second_cpu = list(tensor.clone() for tensor in self._make_case(dtype, seed=47))
        second_cpu[6] = torch.tensor([0, 9, 7, 0], dtype=torch.int32)
        second_cpu[7] = torch.tensor([8, 5, 0, 0], dtype=torch.int32)
        second_cpu = tuple(second_cpu)
        first_expected = _reference_merge(*first_cpu).clone()
        second_expected = _reference_merge(*second_cpu).clone()

        staging = [self._to_npu_base_format(tensor) for tensor in first_cpu]
        output = torch.empty_like(staging[0])
        input_addresses = tuple(tensor.data_ptr() for tensor in staging)
        output_address = output.data_ptr()
        capture_stream = torch.npu.Stream()
        graph_pool = torch.npu.graph_pool_handle()
        torch.npu.synchronize()

        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream):
            for _ in range(2):
                sfa_state_merge_inplace(*staging, output)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            graph, pool=graph_pool, stream=capture_stream, auto_dispatch_capture=True,
        ):
            sfa_state_merge_inplace(*staging, output)
        torch.npu.synchronize()
        self.assertEqual(
            tuple(tensor.data_ptr() for tensor in staging), input_addresses
        )
        self.assertEqual(output.data_ptr(), output_address)

        def replay(case, expected):
            for source, target in zip(case, staging):
                target.copy_(source)
            output.fill_(float("nan"))
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            self.assertEqual(
                tuple(tensor.data_ptr() for tensor in staging), input_addresses
            )
            self.assertEqual(output.data_ptr(), output_address)
            torch.testing.assert_close(
                output.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2
            )

        replay(first_cpu, first_expected)
        replay(second_cpu, second_expected)


if __name__ == "__main__":
    unittest.main()
