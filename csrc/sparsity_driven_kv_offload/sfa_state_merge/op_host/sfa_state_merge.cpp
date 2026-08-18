// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "aclrtlaunch_sfa_state_merge_bf16.h"
#include "aclrtlaunch_sfa_state_merge_fp16.h"

#include <algorithm>
#include <limits>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr uint32_t kTileElements = 1024;
constexpr uint32_t kElementsPerDataBlock = 16;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
    // The raw kernel consumes the dense logical BSND/ND memory order.  Depending
    // on torch_npu version and graph capture, a dense tensor may be tagged as
    // either ACL_FORMAT_ND or ACL_FORMAT_NCHW even though both expose the same
    // contiguous storage for these shapes.  Reject only genuinely non-dense
    // formats; is_contiguous() is checked by the callers below.
    const auto format = at_npu::native::get_npu_format(tensor);
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW, name,
                " must use a dense ACL_FORMAT_ND or ACL_FORMAT_NCHW layout for sfa_state_merge");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as hit_output");
}

void CheckStatsTensor(const at::Tensor &tensor, const at::Tensor &output, const char *name)
{
    CheckNpuTensor(tensor, name);
    CheckSameDevice(tensor, output, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, 1, S, H], got rank ", tensor.dim());
    TORCH_CHECK(tensor.size(0) == output.size(0) && tensor.size(1) == 1 && tensor.size(2) == output.size(1) &&
                    tensor.size(3) == output.size(2),
                name, " must have shape [", output.size(0), ", 1, ", output.size(1), ", ", output.size(2), "], got ",
                tensor.sizes());
}

void CheckCountsTensor(const at::Tensor &tensor, const at::Tensor &output, const char *name)
{
    CheckNpuTensor(tensor, name);
    CheckSameDevice(tensor, output, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == output.size(0), name, " must have shape [", output.size(0),
                "], got ", tensor.sizes());
}

}  // namespace

HOST_API at::Tensor &sfa_state_merge(const at::Tensor &hit_output, const at::Tensor &hit_max,
                                     const at::Tensor &hit_sum, const at::Tensor &miss_output,
                                     const at::Tensor &miss_max, const at::Tensor &miss_sum,
                                     const at::Tensor &hit_counts, const at::Tensor &miss_counts, at::Tensor &output)
{
    CheckNpuTensor(hit_output, "hit_output");
    CheckNpuTensor(miss_output, "miss_output");
    CheckNpuTensor(output, "output");
    CheckSameDevice(miss_output, hit_output, "miss_output");
    CheckSameDevice(output, hit_output, "output");

    TORCH_CHECK(hit_output.is_contiguous(), "hit_output must be contiguous");
    TORCH_CHECK(miss_output.is_contiguous(), "miss_output must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(hit_output.dim() == 4, "hit_output must have shape [B, S, H, D], got rank ", hit_output.dim());
    TORCH_CHECK(hit_output.size(0) > 0 && hit_output.size(1) > 0 && hit_output.size(2) > 0 &&
                    hit_output.size(3) > 0,
                "hit_output B, S, H, and D dimensions must be positive, got ", hit_output.sizes());
    TORCH_CHECK(miss_output.sizes() == hit_output.sizes(), "miss_output shape must match hit_output: ",
                miss_output.sizes(), " vs ", hit_output.sizes());
    TORCH_CHECK(output.sizes() == hit_output.sizes(), "output shape must match hit_output: ", output.sizes(), " vs ",
                hit_output.sizes());

    const at::ScalarType dtype = hit_output.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "hit_output must be float16 or bfloat16, got ", dtype);
    TORCH_CHECK(miss_output.scalar_type() == dtype && output.scalar_type() == dtype,
                "hit_output, miss_output, and output must have the same dtype");

    CheckStatsTensor(hit_max, hit_output, "hit_max");
    CheckStatsTensor(hit_sum, hit_output, "hit_sum");
    CheckStatsTensor(miss_max, hit_output, "miss_max");
    CheckStatsTensor(miss_sum, hit_output, "miss_sum");
    CheckCountsTensor(hit_counts, hit_output, "hit_counts");
    CheckCountsTensor(miss_counts, hit_output, "miss_counts");

    const uint64_t batchSize64 = static_cast<uint64_t>(hit_output.size(0));
    const uint64_t queryLength64 = static_cast<uint64_t>(hit_output.size(1));
    const uint64_t headCount64 = static_cast<uint64_t>(hit_output.size(2));
    const uint64_t headDim64 = static_cast<uint64_t>(hit_output.size(3));
    TORCH_CHECK(batchSize64 <= kUint32Max && queryLength64 <= kUint32Max && headCount64 <= kUint32Max &&
                    headDim64 <= kUint32Max,
                "B, S, H, and D must fit in uint32");
    TORCH_CHECK(headDim64 % 16U == 0,
                "D must be a multiple of 16 for aligned FP16/BF16 vector copies, got ", headDim64);
    const uint64_t outputElements64 = static_cast<uint64_t>(hit_output.numel());
    TORCH_CHECK(outputElements64 <= kUint32Max, "B * S * H * D exceeds the uint32 kernel offset range");
    const uint64_t rowCount64 = outputElements64 / headDim64;
    TORCH_CHECK(rowCount64 <= kUint32Max, "B * S * H exceeds the uint32 row range");

    const uint32_t batchSize = static_cast<uint32_t>(batchSize64);
    const uint32_t queryLength = static_cast<uint32_t>(queryLength64);
    const uint32_t headCount = static_cast<uint32_t>(headCount64);
    const uint32_t headDim = static_cast<uint32_t>(headDim64);
    const uint32_t tileElements =
        headDim >= kTileElements
            ? kTileElements
            : ((headDim + kElementsPerDataBlock - 1U) / kElementsPerDataBlock) * kElementsPerDataBlock;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreCount64 = platform->GetCoreNumAiv();
    TORCH_CHECK(aivCoreCount64 > 0 && static_cast<uint64_t>(aivCoreCount64) <= kUint32Max,
                "AIV core count must be in the uint32 positive range, got ", aivCoreCount64);
    const uint32_t aivCoreCount = static_cast<uint32_t>(aivCoreCount64);
    const uint32_t rowCount = static_cast<uint32_t>(rowCount64);
    const uint32_t blockDim = std::max(1U, std::min(rowCount, aivCoreCount));

    auto npuStream = c10_npu::getCurrentNPUStream();
    hit_output.record_stream(npuStream);
    hit_max.record_stream(npuStream);
    hit_sum.record_stream(npuStream);
    miss_output.record_stream(npuStream);
    miss_max.record_stream(npuStream);
    miss_sum.record_stream(npuStream);
    hit_counts.record_stream(npuStream);
    miss_counts.record_stream(npuStream);
    output.record_stream(npuStream);

    if (dtype == at::kHalf) {
        EXEC_KERNEL_CMD(sfa_state_merge_fp16, blockDim, hit_output, hit_max, hit_sum, miss_output, miss_max, miss_sum,
                        hit_counts, miss_counts, output, batchSize, queryLength, headCount, headDim, tileElements);
    } else {
        EXEC_KERNEL_CMD(sfa_state_merge_bf16, blockDim, hit_output, hit_max, hit_sum, miss_output, miss_max, miss_sum,
                        hit_counts, miss_counts, output, batchSize, queryLength, headCount, headDim, tileElements);
    }
    return output;
}

}  // namespace npu_kernel
}  // namespace sglang
