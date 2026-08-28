/**
 * @file sparse_kv_partition_plan_kernel.cpp
 * @brief Fused fixed-shape planner for sparse KV hit/miss communication.
 */

#include "kernel_operator.h"

constexpr uint32_t FIXED_TOPK = 2048;

class KernelSparseKvPartitionPlan
{
public:
    __aicore__ inline KernelSparseKvPartitionPlan() {}

    __aicore__ inline void Init(
        GM_ADDR tokenOnDevice, GM_ADDR deviceTokenPos, GM_ADDR topkIndices,
        GM_ADDR deviceCacheRowIndices, GM_ADDR slotMapRowIndices, GM_ADDR validTopkMask,
        GM_ADDR hitSparseIndices, GM_ADDR missSparseIndices, GM_ADDR hitCounts, GM_ADDR missCounts,
        GM_ADDR hitSrcIndices, GM_ADDR missSrcIndices, GM_ADDR missHotDstIndices,
        GM_ADDR hitValidMask, GM_ADDR missValidMask, GM_ADDR slotMapFlatIndices,
        GM_ADDR slotMapSlotValues, uint32_t batchSize, uint32_t topk,
        uint32_t maxContextLen, uint32_t slotMapWidth, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->topk = topk;
        this->maxContextLen = maxContextLen;
        this->slotMapWidth = slotMapWidth;

        const uint64_t planSize = static_cast<uint64_t>(batchSize) * topk;
        tokenOnDeviceGm.SetGlobalBuffer((__gm__ int32_t *)tokenOnDevice, planSize);
        deviceTokenPosGm.SetGlobalBuffer((__gm__ int32_t *)deviceTokenPos, planSize);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices, planSize);
        deviceCacheRowIndicesGm.SetGlobalBuffer((__gm__ int64_t *)deviceCacheRowIndices, batchSize);
        slotMapRowIndicesGm.SetGlobalBuffer((__gm__ int64_t *)slotMapRowIndices, batchSize);
        validTopkMaskGm.SetGlobalBuffer((__gm__ uint8_t *)validTopkMask, planSize);

        hitSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)hitSparseIndices, planSize);
        missSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)missSparseIndices, planSize);
        hitCountsGm.SetGlobalBuffer((__gm__ int32_t *)hitCounts, batchSize);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
        hitSrcIndicesGm.SetGlobalBuffer((__gm__ int64_t *)hitSrcIndices, planSize);
        missSrcIndicesGm.SetGlobalBuffer((__gm__ int64_t *)missSrcIndices, planSize);
        missHotDstIndicesGm.SetGlobalBuffer((__gm__ int64_t *)missHotDstIndices, planSize);
        hitValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)hitValidMask, planSize);
        missValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)missValidMask, planSize);
        slotMapFlatIndicesGm.SetGlobalBuffer((__gm__ int64_t *)slotMapFlatIndices, planSize);
        slotMapSlotValuesGm.SetGlobalBuffer((__gm__ int32_t *)slotMapSlotValues, planSize);

        // All fixed-size inputs and outputs are staged through UB. This avoids
        // turning the planner into thousands of 4/8-byte GM transactions.
        pipe->InitBuffer(int32Buf, 7U * FIXED_TOPK * sizeof(int32_t));
        pipe->InitBuffer(maskBuf, 3U * FIXED_TOPK * sizeof(uint8_t));
        pipe->InitBuffer(int64Buf, 4U * FIXED_TOPK * sizeof(int64_t));
        // GlobalTensor::SetValue writes through the per-core DCache. Adjacent
        // batch counters therefore cannot be safely published by different
        // AIVs. Keep scalar metadata in UB and use exact MTE copies instead.
        pipe->InitBuffer(rowMetaBuf, 8U * sizeof(int64_t));
        pipe->InitBuffer(countBuf, 16U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        AscendC::LocalTensor<int32_t> int32Base = int32Buf.Get<int32_t>();
        AscendC::LocalTensor<uint8_t> maskBase = maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<int64_t> int64Base = int64Buf.Get<int64_t>();

        AscendC::LocalTensor<int32_t> tokenOnDevice = int32Base;
        AscendC::LocalTensor<int32_t> deviceTokenPos = int32Base[FIXED_TOPK];
        AscendC::LocalTensor<int32_t> topkIndices = int32Base[2U * FIXED_TOPK];
        AscendC::LocalTensor<int32_t> occupied = int32Base[3U * FIXED_TOPK];
        AscendC::LocalTensor<int32_t> hitSparse = int32Base[4U * FIXED_TOPK];
        AscendC::LocalTensor<int32_t> missSparse = int32Base[5U * FIXED_TOPK];
        AscendC::LocalTensor<int32_t> slotMapValues = int32Base[6U * FIXED_TOPK];

        AscendC::LocalTensor<uint8_t> validTopk = maskBase;
        AscendC::LocalTensor<uint8_t> hitValid = maskBase[FIXED_TOPK];
        AscendC::LocalTensor<uint8_t> missValid = maskBase[2U * FIXED_TOPK];

        AscendC::LocalTensor<int64_t> hitSrc = int64Base;
        AscendC::LocalTensor<int64_t> missSrc = int64Base[FIXED_TOPK];
        AscendC::LocalTensor<int64_t> missHotDst = int64Base[2U * FIXED_TOPK];
        AscendC::LocalTensor<int64_t> slotMapFlat = int64Base[3U * FIXED_TOPK];
        AscendC::LocalTensor<int64_t> rowMeta = rowMetaBuf.Get<int64_t>();
        AscendC::LocalTensor<int32_t> counts = countBuf.Get<int32_t>();

        for (uint32_t batch = workerIndex; batch < batchSize; batch += workerCount) {
            BuildOneBatch(batch, tokenOnDevice, deviceTokenPos, topkIndices,
                          occupied, hitSparse, missSparse, slotMapValues,
                          validTopk, hitValid, missValid, hitSrc, missSrc,
                          missHotDst, slotMapFlat, rowMeta, counts);
        }
    }

private:
    __aicore__ inline void BuildOneBatch(
        uint32_t batch, AscendC::LocalTensor<int32_t> tokenOnDevice,
        AscendC::LocalTensor<int32_t> deviceTokenPos,
        AscendC::LocalTensor<int32_t> topkIndices,
        AscendC::LocalTensor<int32_t> occupied,
        AscendC::LocalTensor<int32_t> hitSparse,
        AscendC::LocalTensor<int32_t> missSparse,
        AscendC::LocalTensor<int32_t> slotMapValues,
        AscendC::LocalTensor<uint8_t> validTopk,
        AscendC::LocalTensor<uint8_t> hitValid,
        AscendC::LocalTensor<uint8_t> missValid,
        AscendC::LocalTensor<int64_t> hitSrc,
        AscendC::LocalTensor<int64_t> missSrc,
        AscendC::LocalTensor<int64_t> missHotDst,
        AscendC::LocalTensor<int64_t> slotMapFlat,
        AscendC::LocalTensor<int64_t> rowMeta,
        AscendC::LocalTensor<int32_t> counts)
    {
        const uint32_t rowOffset = batch * topk;
        AscendC::DataCopy(tokenOnDevice, tokenOnDeviceGm[rowOffset], topk);
        AscendC::DataCopy(deviceTokenPos, deviceTokenPosGm[rowOffset], topk);
        AscendC::DataCopy(topkIndices, topkIndicesGm[rowOffset], topk);
        AscendC::DataCopy(validTopk, validTopkMaskGm[rowOffset], topk);
        AscendC::DataCopyExtParams rowCopyParams{
            1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> rowPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(rowMeta, deviceCacheRowIndicesGm[batch],
                             rowCopyParams, rowPadParams);
        AscendC::DataCopyPad(rowMeta[4], slotMapRowIndicesGm[batch],
                             rowCopyParams, rowPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

        AscendC::Duplicate(occupied, static_cast<int32_t>(0), FIXED_TOPK);
        AscendC::Duplicate(hitSparse, static_cast<int32_t>(-1), FIXED_TOPK);
        AscendC::Duplicate(missSparse, static_cast<int32_t>(-1), FIXED_TOPK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        const int64_t cacheRow = rowMeta.GetValue(0);
        const int64_t slotMapRow = rowMeta.GetValue(4);
        const int64_t cacheBase = cacheRow * static_cast<int64_t>(topk);
        const int64_t hostBase = cacheRow * static_cast<int64_t>(maxContextLen);
        const int64_t slotMapBase = slotMapRow * static_cast<int64_t>(slotMapWidth);

        uint32_t hitCount = 0;
        uint32_t missCount = 0;
        for (uint32_t position = 0; position < topk; ++position) {
            const bool valid = validTopk.GetValue(position) != 0;
            const int32_t logicalToken = topkIndices.GetValue(position);
            const int32_t rawSlot = deviceTokenPos.GetValue(position);
            const bool hit = valid && tokenOnDevice.GetValue(position) != 0 && rawSlot >= 0 &&
                             static_cast<uint32_t>(rawSlot) < topk;
            const bool miss = valid && !hit;

            hitValid.SetValue(position, hit ? 1U : 0U);
            missValid.SetValue(position, miss ? 1U : 0U);

            const int32_t safeSlot = hit ? rawSlot : 0;
            int32_t safeToken = logicalToken;
            if (safeToken < 0) {
                safeToken = 0;
            } else if (static_cast<uint32_t>(safeToken) >= maxContextLen) {
                safeToken = static_cast<int32_t>(maxContextLen - 1U);
            }
            hitSrc.SetValue(position, cacheBase + static_cast<int64_t>(safeSlot));
            missSrc.SetValue(position, hostBase + static_cast<int64_t>(safeToken));
            missHotDst.SetValue(position, cacheBase);

            const int64_t publishToken = valid ? static_cast<int64_t>(logicalToken)
                                               : static_cast<int64_t>(maxContextLen);
            slotMapFlat.SetValue(position, slotMapBase + publishToken);
            slotMapValues.SetValue(position, hit ? rawSlot : -1);

            if (hit) {
                occupied.SetValue(static_cast<uint32_t>(rawSlot), 1);
                hitSparse.SetValue(hitCount++, static_cast<int32_t>(position));
            } else if (miss) {
                missSparse.SetValue(missCount++, static_cast<int32_t>(position));
            }
        }

        if (hitCount == 0) {
            hitSparse.SetValue(0, 0);
        }
        if (missCount == 0) {
            missSparse.SetValue(0, 0);
        }
        counts.SetValue(0, static_cast<int32_t>(hitCount));
        counts.SetValue(8, static_cast<int32_t>(missCount));

        // Preserve hit slots. Assign misses, in Top-K order, to the ascending
        // complement of those occupied slots. This is deterministic across
        // graph replay and never overwrites a KV row still selected as a hit.
        uint32_t nextFreeSlot = 0;
        for (uint32_t position = 0; position < topk; ++position) {
            const bool valid = validTopk.GetValue(position) != 0;
            const int32_t rawSlot = deviceTokenPos.GetValue(position);
            const bool hit = valid && tokenOnDevice.GetValue(position) != 0 && rawSlot >= 0 &&
                             static_cast<uint32_t>(rawSlot) < topk;
            if (!valid || hit) {
                continue;
            }
            while (nextFreeSlot < topk && occupied.GetValue(nextFreeSlot) != 0) {
                ++nextFreeSlot;
            }
            if (nextFreeSlot >= topk) {
                // This cannot happen for a K-entry plan, but keep every output
                // descriptor in bounds if malformed duplicated input arrives.
                missValid.SetValue(position, 0U);
                slotMapValues.SetValue(position, -1);
                continue;
            }
            missHotDst.SetValue(position, cacheBase + static_cast<int64_t>(nextFreeSlot));
            slotMapValues.SetValue(position, static_cast<int32_t>(nextFreeSlot));
            ++nextFreeSlot;
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::DataCopyExtParams countCopyParams{
            1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(hitCountsGm[batch], counts, countCopyParams);
        AscendC::DataCopyPad(missCountsGm[batch], counts[8], countCopyParams);
        AscendC::DataCopy(hitSparseIndicesGm[rowOffset], hitSparse, topk);
        AscendC::DataCopy(missSparseIndicesGm[rowOffset], missSparse, topk);
        AscendC::DataCopy(hitSrcIndicesGm[rowOffset], hitSrc, topk);
        AscendC::DataCopy(missSrcIndicesGm[rowOffset], missSrc, topk);
        AscendC::DataCopy(missHotDstIndicesGm[rowOffset], missHotDst, topk);
        AscendC::DataCopy(hitValidMaskGm[rowOffset], hitValid, topk);
        AscendC::DataCopy(missValidMaskGm[rowOffset], missValid, topk);
        AscendC::DataCopy(slotMapFlatIndicesGm[rowOffset], slotMapFlat, topk);
        AscendC::DataCopy(slotMapSlotValuesGm[rowOffset], slotMapValues, topk);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }

    AscendC::GlobalTensor<int32_t> tokenOnDeviceGm;
    AscendC::GlobalTensor<int32_t> deviceTokenPosGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int64_t> deviceCacheRowIndicesGm;
    AscendC::GlobalTensor<int64_t> slotMapRowIndicesGm;
    AscendC::GlobalTensor<uint8_t> validTopkMaskGm;
    AscendC::GlobalTensor<int32_t> hitSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> missSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> hitCountsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;
    AscendC::GlobalTensor<int64_t> hitSrcIndicesGm;
    AscendC::GlobalTensor<int64_t> missSrcIndicesGm;
    AscendC::GlobalTensor<int64_t> missHotDstIndicesGm;
    AscendC::GlobalTensor<uint8_t> hitValidMaskGm;
    AscendC::GlobalTensor<uint8_t> missValidMaskGm;
    AscendC::GlobalTensor<int64_t> slotMapFlatIndicesGm;
    AscendC::GlobalTensor<int32_t> slotMapSlotValuesGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int32Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int64Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowMetaBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> countBuf;
    uint32_t batchSize = 0;
    uint32_t topk = 0;
    uint32_t maxContextLen = 0;
    uint32_t slotMapWidth = 0;
};

extern "C" __global__ __aicore__ void sparse_kv_partition_plan(
    GM_ADDR token_on_device, GM_ADDR device_token_pos, GM_ADDR topk_indices,
    GM_ADDR device_cache_row_indices, GM_ADDR slot_map_row_indices, GM_ADDR valid_topk_mask,
    GM_ADDR hit_sparse_indices, GM_ADDR miss_sparse_indices, GM_ADDR hit_counts, GM_ADDR miss_counts,
    GM_ADDR hit_src_indices, GM_ADDR miss_src_indices, GM_ADDR miss_hot_dst_indices,
    GM_ADDR hit_valid_mask, GM_ADDR miss_valid_mask, GM_ADDR slot_map_flat_indices,
    GM_ADDR slot_map_slot_values, uint32_t batch_size, uint32_t topk,
    uint32_t max_context_len, uint32_t slot_map_width)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseKvPartitionPlan kernel;
    kernel.Init(token_on_device, device_token_pos, topk_indices, device_cache_row_indices,
                slot_map_row_indices, valid_topk_mask, hit_sparse_indices, miss_sparse_indices,
                hit_counts, miss_counts, hit_src_indices, miss_src_indices, miss_hot_dst_indices,
                hit_valid_mask, miss_valid_mask, slot_map_flat_indices, slot_map_slot_values,
                batch_size, topk, max_context_len, slot_map_width, &pipe);
    kernel.Process();
}
