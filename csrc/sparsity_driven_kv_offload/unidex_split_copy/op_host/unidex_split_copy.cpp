// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.

#include "defines.h"
#include "torch_helper.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "aclrtlaunch_unidex_split_copy.h"
#include "aclrtlaunch_unidex_split_copy_promote.h"

#include <cstdint>
#include <limits>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t kMaxRowBytes = 32 * 1024;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
    const auto format = at_npu::native::get_npu_format(tensor);
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW, name,
                " must use a dense ACL_FORMAT_ND or ACL_FORMAT_NCHW layout");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as dst_nope");
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max, name, " exceeds uint32 range: ", value);
}

void CheckCommon(const at::Tensor &src, at::Tensor &dst_nope, at::Tensor &dst_rope,
                 const at::Tensor &src_index, const at::Tensor &dst_index, const at::Tensor &valid_mask,
                 int64_t src_rows, int64_t dst_rows, int64_t nope_bytes, int64_t rope_bytes, int64_t max_copy,
                 int64_t block_dim, const c10::optional<int64_t> &src_ptr)
{
    const bool useRawSrc = src_ptr.has_value();
    CheckNpuTensor(dst_nope, "dst_nope");
    CheckNpuTensor(dst_rope, "dst_rope");
    CheckNpuTensor(src_index, "src_index");
    CheckNpuTensor(dst_index, "dst_index");
    CheckNpuTensor(valid_mask, "valid_mask");
    CheckSameDevice(dst_rope, dst_nope, "dst_rope");
    CheckSameDevice(src_index, dst_nope, "src_index");
    CheckSameDevice(dst_index, dst_nope, "dst_index");
    CheckSameDevice(valid_mask, dst_nope, "valid_mask");
    if (!useRawSrc) {
        CheckNpuTensor(src, "src");
        CheckSameDevice(src, dst_nope, "src");
    }

    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst_nope.is_contiguous(), "dst_nope must be contiguous");
    TORCH_CHECK(dst_rope.is_contiguous(), "dst_rope must be contiguous");
    TORCH_CHECK(src_index.is_contiguous(), "src_index must be contiguous");
    TORCH_CHECK(dst_index.is_contiguous(), "dst_index must be contiguous");
    TORCH_CHECK(valid_mask.is_contiguous(), "valid_mask must be contiguous");
    TORCH_CHECK(src.scalar_type() == dst_nope.scalar_type() && src.scalar_type() == dst_rope.scalar_type(),
                "src, dst_nope, and dst_rope must have the same dtype");
    TORCH_CHECK(src.scalar_type() == at::kHalf || src.scalar_type() == at::kBFloat16,
                "split copy supports float16 and bfloat16, got ", src.scalar_type());
    TORCH_CHECK(src_index.scalar_type() == at::kLong, "src_index must be int64, got ", src_index.scalar_type());
    TORCH_CHECK(dst_index.scalar_type() == at::kLong, "dst_index must be int64, got ", dst_index.scalar_type());
    TORCH_CHECK(valid_mask.scalar_type() == at::kBool || valid_mask.scalar_type() == at::kByte,
                "valid_mask must be bool or uint8, got ", valid_mask.scalar_type());
    TORCH_CHECK(src_index.dim() == 1 && dst_index.dim() == 1 && valid_mask.dim() == 1,
                "src_index, dst_index, and valid_mask must be 1-D");

    TORCH_CHECK(!useRawSrc || *src_ptr > 0, "src_ptr must be a non-zero address");
    TORCH_CHECK(src_rows > 0 && dst_rows > 0, "src_rows and dst_rows must be positive");
    TORCH_CHECK(nope_bytes > 0 && rope_bytes > 0, "nope_bytes and rope_bytes must be positive");
    TORCH_CHECK(nope_bytes + rope_bytes <= kMaxRowBytes,
                "combined KV row exceeds the supported 32 KiB limit");
    TORCH_CHECK(max_copy >= 0, "max_copy must be non-negative, got ", max_copy);
    TORCH_CHECK(block_dim > 0, "block_dim must be positive, got ", block_dim);
    TORCH_CHECK(nope_bytes % 32 == 0 && rope_bytes % 32 == 0,
                "NoPE and RoPE row byte sizes must be 32-byte aligned, got ", nope_bytes, " and ", rope_bytes);

    CheckFitsUint32(src_rows, "src_rows");
    CheckFitsUint32(dst_rows, "dst_rows");
    CheckFitsUint32(nope_bytes, "nope_bytes");
    CheckFitsUint32(rope_bytes, "rope_bytes");
    CheckFitsUint32(max_copy, "max_copy");
    CheckFitsUint32(block_dim, "block_dim");
    TORCH_CHECK(src_index.numel() >= max_copy && dst_index.numel() >= max_copy &&
                    valid_mask.numel() >= max_copy,
                "copy descriptors have fewer elements than max_copy=", max_copy);

    const uint64_t rowBytes = static_cast<uint64_t>(nope_bytes + rope_bytes);
    const uint64_t srcRequiredBytes = static_cast<uint64_t>(src_rows) * rowBytes;
    const uint64_t nopeRequiredBytes = static_cast<uint64_t>(dst_rows) * static_cast<uint64_t>(nope_bytes);
    const uint64_t ropeRequiredBytes = static_cast<uint64_t>(dst_rows) * static_cast<uint64_t>(rope_bytes);
    TORCH_CHECK(srcRequiredBytes <= kUint32Max && nopeRequiredBytes <= kUint32Max && ropeRequiredBytes <= kUint32Max,
                "split-copy tensor extent exceeds the kernel uint32 address range");
    if (!useRawSrc) {
        const uint64_t srcAvailableBytes =
            static_cast<uint64_t>(src.numel()) * static_cast<uint64_t>(src.element_size());
        TORCH_CHECK(srcRequiredBytes <= srcAvailableBytes, "src storage is too small");
    }
    const uint64_t nopeAvailableBytes =
        static_cast<uint64_t>(dst_nope.numel()) * static_cast<uint64_t>(dst_nope.element_size());
    const uint64_t ropeAvailableBytes =
        static_cast<uint64_t>(dst_rope.numel()) * static_cast<uint64_t>(dst_rope.element_size());
    TORCH_CHECK(nopeRequiredBytes <= nopeAvailableBytes, "dst_nope storage is too small");
    TORCH_CHECK(ropeRequiredBytes <= ropeAvailableBytes, "dst_rope storage is too small");
}

void RecordCommon(const at::Tensor &src, at::Tensor &dst_nope, at::Tensor &dst_rope,
                  const at::Tensor &src_index, const at::Tensor &dst_index, const at::Tensor &valid_mask,
                  bool useRawSrc, c10_npu::NPUStream &stream)
{
    if (!useRawSrc) {
        src.record_stream(stream);
    }
    dst_nope.record_stream(stream);
    dst_rope.record_stream(stream);
    src_index.record_stream(stream);
    dst_index.record_stream(stream);
    valid_mask.record_stream(stream);
}

}  // namespace

HOST_API void unidex_split_copy(const at::Tensor &src, at::Tensor &dst_nope, at::Tensor &dst_rope,
                                const at::Tensor &src_index, const at::Tensor &dst_index,
                                const at::Tensor &valid_mask, int64_t src_rows, int64_t dst_rows,
                                int64_t nope_bytes, int64_t rope_bytes, int64_t max_copy, int64_t block_dim,
                                c10::optional<int64_t> src_ptr)
{
    CheckCommon(src, dst_nope, dst_rope, src_index, dst_index, valid_mask, src_rows, dst_rows, nope_bytes,
                rope_bytes, max_copy, block_dim, src_ptr);
    if (max_copy == 0) {
        return;
    }

    auto npuStream = c10_npu::getCurrentNPUStream();
    RecordCommon(src, dst_nope, dst_rope, src_index, dst_index, valid_mask, src_ptr.has_value(), npuStream);
    void *srcAddr = src_ptr.has_value() ? reinterpret_cast<void *>(static_cast<uintptr_t>(*src_ptr))
                                       : const_cast<void *>(src.data_ptr());
    EXEC_KERNEL_CMD(unidex_split_copy, static_cast<uint32_t>(block_dim), srcAddr, dst_nope, dst_rope, src_index,
                    dst_index, valid_mask, static_cast<uint32_t>(src_rows), static_cast<uint32_t>(dst_rows),
                    static_cast<uint32_t>(nope_bytes), static_cast<uint32_t>(rope_bytes),
                    static_cast<uint32_t>(max_copy));
}

HOST_API void unidex_split_copy_promote(const at::Tensor &src, at::Tensor &dst_nope, at::Tensor &dst_rope,
                                        at::Tensor &hot_cache, const at::Tensor &src_index,
                                        const at::Tensor &dst_index, const at::Tensor &hot_dst_index,
                                        const at::Tensor &valid_mask, int64_t src_rows, int64_t dst_rows,
                                        int64_t hot_rows, int64_t nope_bytes, int64_t rope_bytes, int64_t max_copy,
                                        int64_t block_dim, c10::optional<int64_t> src_ptr)
{
    CheckCommon(src, dst_nope, dst_rope, src_index, dst_index, valid_mask, src_rows, dst_rows, nope_bytes,
                rope_bytes, max_copy, block_dim, src_ptr);
    CheckNpuTensor(hot_cache, "hot_cache");
    CheckNpuTensor(hot_dst_index, "hot_dst_index");
    CheckSameDevice(hot_cache, dst_nope, "hot_cache");
    CheckSameDevice(hot_dst_index, dst_nope, "hot_dst_index");
    TORCH_CHECK(hot_cache.is_contiguous(), "hot_cache must be contiguous");
    TORCH_CHECK(hot_dst_index.is_contiguous() && hot_dst_index.dim() == 1,
                "hot_dst_index must be contiguous and 1-D");
    TORCH_CHECK(hot_dst_index.scalar_type() == at::kLong, "hot_dst_index must be int64");
    TORCH_CHECK(hot_cache.scalar_type() == src.scalar_type(), "hot_cache dtype must match src");
    TORCH_CHECK(hot_rows > 0, "hot_rows must be positive, got ", hot_rows);
    CheckFitsUint32(hot_rows, "hot_rows");
    TORCH_CHECK(hot_dst_index.numel() >= max_copy, "hot_dst_index has fewer elements than max_copy");
    const uint64_t hotRequiredBytes = static_cast<uint64_t>(hot_rows) *
                                      static_cast<uint64_t>(nope_bytes + rope_bytes);
    const uint64_t hotAvailableBytes =
        static_cast<uint64_t>(hot_cache.numel()) * static_cast<uint64_t>(hot_cache.element_size());
    TORCH_CHECK(hotRequiredBytes <= kUint32Max, "hot-cache extent exceeds the kernel uint32 address range");
    TORCH_CHECK(hotRequiredBytes <= hotAvailableBytes, "hot_cache storage is too small");
    if (max_copy == 0) {
        return;
    }

    auto npuStream = c10_npu::getCurrentNPUStream();
    RecordCommon(src, dst_nope, dst_rope, src_index, dst_index, valid_mask, src_ptr.has_value(), npuStream);
    hot_cache.record_stream(npuStream);
    hot_dst_index.record_stream(npuStream);
    void *srcAddr = src_ptr.has_value() ? reinterpret_cast<void *>(static_cast<uintptr_t>(*src_ptr))
                                       : const_cast<void *>(src.data_ptr());
    EXEC_KERNEL_CMD(unidex_split_copy_promote, static_cast<uint32_t>(block_dim), srcAddr, dst_nope, dst_rope,
                    hot_cache, src_index, dst_index, hot_dst_index, valid_mask, static_cast<uint32_t>(src_rows),
                    static_cast<uint32_t>(dst_rows), static_cast<uint32_t>(hot_rows),
                    static_cast<uint32_t>(nope_bytes), static_cast<uint32_t>(rope_bytes),
                    static_cast<uint32_t>(max_copy));
}

}  // namespace npu_kernel
}  // namespace sglang
