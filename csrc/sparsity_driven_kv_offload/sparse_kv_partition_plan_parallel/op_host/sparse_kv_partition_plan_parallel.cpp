// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");

#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_sparse_kv_partition_classify.h"
#include "aclrtlaunch_sparse_kv_partition_scan.h"
#include "aclrtlaunch_sparse_kv_partition_scatter.h"

#include <algorithm>
#include <limits>

namespace sglang {
namespace npu_kernel {
namespace {

constexpr uint32_t kFixedTopk = 2048;
constexpr uint32_t kTileLen = 64;
constexpr uint32_t kTilesPerBatch = kFixedTopk / kTileLen;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1,
                name, " must be on an NPU device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference,
                     const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name,
                " must be on the same NPU device");
}

void CheckTensor(const at::Tensor &tensor, const at::Tensor &reference,
                 const char *name, at::ScalarType dtype, int64_t expectedNumel)
{
    CheckNpuTensor(tensor, name);
    CheckSameDevice(tensor, reference, name);
    TORCH_CHECK(tensor.scalar_type() == dtype, name,
                " has wrong dtype: ", tensor.scalar_type());
    TORCH_CHECK(tensor.numel() == expectedNumel, name, " must have ",
                expectedNumel, " elements, got ", tensor.numel());
}

}  // namespace

HOST_API void sparse_kv_partition_plan_parallel(
    const at::Tensor &token_on_device, const at::Tensor &device_token_pos,
    const at::Tensor &topk_indices,
    const at::Tensor &device_cache_row_indices,
    const at::Tensor &slot_map_row_indices,
    const at::Tensor &valid_topk_mask, at::Tensor &hit_sparse_indices,
    at::Tensor &miss_sparse_indices, at::Tensor &hit_counts,
    at::Tensor &miss_counts, at::Tensor &hit_src_indices,
    at::Tensor &miss_src_indices, at::Tensor &miss_hot_dst_indices,
    at::Tensor &hit_valid_mask, at::Tensor &miss_valid_mask,
    at::Tensor &slot_map_flat_indices, at::Tensor &slot_map_slot_values,
    at::Tensor &tile_hit_counts, at::Tensor &tile_miss_counts,
    at::Tensor &tile_hit_offsets, at::Tensor &tile_miss_offsets,
    at::Tensor &selected_slots, at::Tensor &tile_occupied_bitmaps,
    at::Tensor &occupied_bitmaps, at::Tensor &free_slot_prefixes,
    int64_t max_context_len,
    int64_t slot_map_width, int64_t block_dim,
    bool output_physical_slots)
{
    CheckNpuTensor(token_on_device, "token_on_device");
    TORCH_CHECK(token_on_device.dim() == 2,
                "token_on_device must have shape [B, K]");
    TORCH_CHECK(token_on_device.scalar_type() == at::kInt,
                "token_on_device must be int32");

    const int64_t batchSize64 = token_on_device.size(0);
    const int64_t topk64 = token_on_device.size(1);
    TORCH_CHECK(batchSize64 > 0, "batch size must be positive");
    TORCH_CHECK(topk64 == kFixedTopk,
                "sparse_kv_partition_plan_parallel requires K=", kFixedTopk,
                ", got ", topk64);
    TORCH_CHECK(max_context_len > 0 &&
                    max_context_len < static_cast<int64_t>(kUint32Max),
                "max_context_len is out of range: ", max_context_len);
    TORCH_CHECK(slot_map_width > max_context_len &&
                    slot_map_width < static_cast<int64_t>(kUint32Max),
                "slot_map_width must include a sentinel column beyond "
                "max_context_len, got width=", slot_map_width,
                " max_context_len=", max_context_len);
    TORCH_CHECK(batchSize64 * topk64 <= static_cast<int64_t>(kUint32Max),
                "B * K exceeds uint32 range");
    TORCH_CHECK(block_dim >= 0 &&
                    block_dim <= static_cast<int64_t>(kUint32Max),
                "block_dim is out of range: ", block_dim);

    const int64_t planElements = batchSize64 * topk64;
    const int64_t tileElements = batchSize64 * kTilesPerBatch;
    CheckTensor(device_token_pos, token_on_device, "device_token_pos", at::kInt,
                planElements);
    CheckTensor(topk_indices, token_on_device, "topk_indices", at::kInt,
                planElements);
    CheckTensor(device_cache_row_indices, token_on_device,
                "device_cache_row_indices", at::kLong, batchSize64);
    CheckTensor(slot_map_row_indices, token_on_device, "slot_map_row_indices",
                at::kLong, batchSize64);
    CheckTensor(valid_topk_mask, token_on_device, "valid_topk_mask", at::kBool,
                planElements);
    CheckTensor(hit_sparse_indices, token_on_device, "hit_sparse_indices", at::kInt,
                planElements);
    CheckTensor(miss_sparse_indices, token_on_device, "miss_sparse_indices", at::kInt,
                planElements);
    CheckTensor(hit_counts, token_on_device, "hit_counts", at::kInt, batchSize64);
    CheckTensor(miss_counts, token_on_device, "miss_counts", at::kInt, batchSize64);
    CheckTensor(hit_src_indices, token_on_device, "hit_src_indices", at::kLong,
                planElements);
    CheckTensor(miss_src_indices, token_on_device, "miss_src_indices", at::kLong,
                planElements);
    CheckTensor(miss_hot_dst_indices, token_on_device, "miss_hot_dst_indices",
                at::kLong, planElements);
    CheckTensor(hit_valid_mask, token_on_device, "hit_valid_mask", at::kBool,
                planElements);
    CheckTensor(miss_valid_mask, token_on_device, "miss_valid_mask", at::kBool,
                planElements);
    CheckTensor(slot_map_flat_indices, token_on_device, "slot_map_flat_indices",
                at::kLong, planElements);
    CheckTensor(slot_map_slot_values, token_on_device, "slot_map_slot_values",
                at::kInt, planElements);
    CheckTensor(tile_hit_counts, token_on_device, "tile_hit_counts", at::kInt,
                tileElements);
    CheckTensor(tile_miss_counts, token_on_device, "tile_miss_counts", at::kInt,
                tileElements);
    CheckTensor(tile_hit_offsets, token_on_device, "tile_hit_offsets", at::kInt,
                tileElements);
    CheckTensor(tile_miss_offsets, token_on_device, "tile_miss_offsets", at::kInt,
                tileElements);
    CheckTensor(selected_slots, token_on_device, "selected_slots", at::kInt,
                planElements);
    CheckTensor(tile_occupied_bitmaps, token_on_device,
                "tile_occupied_bitmaps", at::kInt,
                planElements);
    CheckTensor(occupied_bitmaps, token_on_device, "occupied_bitmaps",
                at::kInt, batchSize64 * 64);
    CheckTensor(free_slot_prefixes, token_on_device, "free_slot_prefixes",
                at::kInt, batchSize64 * 64);

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t maxAiv = static_cast<uint32_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(maxAiv > 0, "failed to get the available AIV core count");
    const uint32_t batchSize = static_cast<uint32_t>(batchSize64);
    const uint32_t requestedBlockDim = static_cast<uint32_t>(block_dim);
    TORCH_CHECK(requestedBlockDim == 0 || requestedBlockDim <= maxAiv,
                "block_dim must not exceed the available AIV core count ",
                maxAiv, ", got ", requestedBlockDim);
    const uint32_t tileBlockDim = requestedBlockDim == 0 ? maxAiv
                                                         : requestedBlockDim;
    const uint32_t scanBlockDim = requestedBlockDim == 0
                                      ? std::min(maxAiv, batchSize)
                                      : std::min(tileBlockDim, batchSize);
    TORCH_CHECK(tileBlockDim > 0 && scanBlockDim > 0,
                "parallel partition planner requires positive block dimensions");

    auto stream = c10_npu::getCurrentNPUStream();
    token_on_device.record_stream(stream);
    device_token_pos.record_stream(stream);
    topk_indices.record_stream(stream);
    device_cache_row_indices.record_stream(stream);
    slot_map_row_indices.record_stream(stream);
    valid_topk_mask.record_stream(stream);
    hit_sparse_indices.record_stream(stream);
    miss_sparse_indices.record_stream(stream);
    hit_counts.record_stream(stream);
    miss_counts.record_stream(stream);
    hit_src_indices.record_stream(stream);
    miss_src_indices.record_stream(stream);
    miss_hot_dst_indices.record_stream(stream);
    hit_valid_mask.record_stream(stream);
    miss_valid_mask.record_stream(stream);
    slot_map_flat_indices.record_stream(stream);
    slot_map_slot_values.record_stream(stream);
    tile_hit_counts.record_stream(stream);
    tile_miss_counts.record_stream(stream);
    tile_hit_offsets.record_stream(stream);
    tile_miss_offsets.record_stream(stream);
    selected_slots.record_stream(stream);
    tile_occupied_bitmaps.record_stream(stream);
    occupied_bitmaps.record_stream(stream);
    free_slot_prefixes.record_stream(stream);

    const uint32_t topk = static_cast<uint32_t>(topk64);
    const uint32_t maxContextLen = static_cast<uint32_t>(max_context_len);
    const uint32_t slotMapWidth = static_cast<uint32_t>(slot_map_width);
    const uint32_t outputPhysicalSlots = static_cast<uint32_t>(output_physical_slots);
    EXEC_KERNEL_CMD(
        sparse_kv_partition_classify, tileBlockDim, token_on_device,
        device_token_pos, topk_indices, device_cache_row_indices,
        slot_map_row_indices, valid_topk_mask, hit_sparse_indices,
        miss_sparse_indices, hit_src_indices, miss_src_indices,
        hit_valid_mask, miss_valid_mask, slot_map_flat_indices,
        tile_hit_counts, tile_miss_counts, selected_slots,
        tile_occupied_bitmaps, batchSize, topk, maxContextLen,
        slotMapWidth);
    EXEC_KERNEL_CMD(
        sparse_kv_partition_scan, scanBlockDim, hit_sparse_indices,
        miss_sparse_indices, hit_counts, miss_counts, tile_hit_counts,
        tile_miss_counts, tile_hit_offsets, tile_miss_offsets,
        tile_occupied_bitmaps, occupied_bitmaps, free_slot_prefixes,
        batchSize, topk);
    EXEC_KERNEL_CMD(
        sparse_kv_partition_scatter, tileBlockDim,
        device_cache_row_indices, tile_hit_counts, tile_miss_counts,
        tile_hit_offsets,
        tile_miss_offsets, selected_slots, occupied_bitmaps,
        free_slot_prefixes, hit_sparse_indices, miss_sparse_indices,
        miss_hot_dst_indices, slot_map_slot_values, batchSize, topk,
        outputPhysicalSlots);
}

}  // namespace npu_kernel
}  // namespace sglang
