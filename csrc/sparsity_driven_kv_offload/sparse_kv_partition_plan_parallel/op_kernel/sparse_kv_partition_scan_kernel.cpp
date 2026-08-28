/**
 * @file sparse_kv_partition_scan_kernel.cpp
 * @brief Request-level tile scan and miss-slot planning.
 */

#include "kernel_operator.h"

constexpr uint32_t FIXED_TOPK = 2048;
constexpr uint32_t TILES_PER_BATCH = 32;
constexpr uint32_t SLOT_BITMAP_WORDS = FIXED_TOPK / 32;

class KernelSparseKvPartitionScan
{
public:
    __aicore__ inline KernelSparseKvPartitionScan() {}

    __aicore__ inline void Init(
        GM_ADDR deviceCacheRowIndices, GM_ADDR hitSparseIndices, GM_ADDR missSparseIndices,
        GM_ADDR hitCounts, GM_ADDR missCounts, GM_ADDR missHotDstIndices,
        GM_ADDR hitValidMask, GM_ADDR missValidMask, GM_ADDR slotMapSlotValues,
        GM_ADDR tileHitCounts, GM_ADDR tileMissCounts, GM_ADDR tileHitOffsets,
        GM_ADDR tileMissOffsets, GM_ADDR selectedSlots, uint32_t batchSize,
        uint32_t topk, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->topk = topk;
        const uint64_t planSize = static_cast<uint64_t>(batchSize) * topk;
        const uint64_t tileSize = static_cast<uint64_t>(batchSize) * TILES_PER_BATCH;
        deviceCacheRowIndicesGm.SetGlobalBuffer((__gm__ int64_t *)deviceCacheRowIndices, batchSize);
        hitSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)hitSparseIndices, planSize);
        missSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)missSparseIndices, planSize);
        hitCountsGm.SetGlobalBuffer((__gm__ int32_t *)hitCounts, batchSize);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
        missHotDstIndicesGm.SetGlobalBuffer((__gm__ int64_t *)missHotDstIndices, planSize);
        hitValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)hitValidMask, planSize);
        missValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)missValidMask, planSize);
        slotMapSlotValuesGm.SetGlobalBuffer((__gm__ int32_t *)slotMapSlotValues, planSize);
        tileHitCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileHitCounts, tileSize);
        tileMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileMissCounts, tileSize);
        tileHitOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)tileHitOffsets, tileSize);
        tileMissOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)tileMissOffsets, tileSize);
        selectedSlotsGm.SetGlobalBuffer((__gm__ int32_t *)selectedSlots, planSize);
        pipe->InitBuffer(occupiedBuf, SLOT_BITMAP_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(maskBuf, 2U * FIXED_TOPK * sizeof(uint8_t));
        pipe->InitBuffer(int32Buf, 2U * FIXED_TOPK * sizeof(int32_t));
        pipe->InitBuffer(int64Buf, FIXED_TOPK * sizeof(int64_t));
        // Four 32-entry tile rows plus scalar staging. All metadata crosses
        // kernel boundaries through MTE, not GlobalTensor scalar DCache.
        pipe->InitBuffer(tileMetaBuf, 5U * TILES_PER_BATCH * sizeof(int32_t));
        pipe->InitBuffer(rowMetaBuf, 4U * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        AscendC::LocalTensor<uint32_t> occupied = occupiedBuf.Get<uint32_t>();
        AscendC::LocalTensor<uint8_t> maskBase = maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> hitValid = maskBase;
        AscendC::LocalTensor<uint8_t> missValid = maskBase[FIXED_TOPK];
        AscendC::LocalTensor<int32_t> int32Base = int32Buf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> selectedSlots = int32Base;
        AscendC::LocalTensor<int32_t> slotMapValues = int32Base[FIXED_TOPK];
        AscendC::LocalTensor<int64_t> missHotDst = int64Buf.Get<int64_t>();
        AscendC::LocalTensor<int32_t> tileMeta = tileMetaBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> tileHitCounts = tileMeta;
        AscendC::LocalTensor<int32_t> tileMissCounts = tileMeta[TILES_PER_BATCH];
        AscendC::LocalTensor<int32_t> tileHitOffsets = tileMeta[2U * TILES_PER_BATCH];
        AscendC::LocalTensor<int32_t> tileMissOffsets = tileMeta[3U * TILES_PER_BATCH];
        AscendC::LocalTensor<int32_t> scalarMeta = tileMeta[4U * TILES_PER_BATCH];
        AscendC::LocalTensor<int64_t> rowMeta = rowMetaBuf.Get<int64_t>();

        for (uint32_t batch = workerIndex; batch < batchSize; batch += workerCount) {
            const uint32_t rowOffset = batch * topk;
            const uint32_t tileOffset = batch * TILES_PER_BATCH;
            AscendC::DataCopy(tileHitCounts, tileHitCountsGm[tileOffset],
                              TILES_PER_BATCH);
            AscendC::DataCopy(tileMissCounts, tileMissCountsGm[tileOffset],
                              TILES_PER_BATCH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

            uint32_t hitTotal = 0;
            uint32_t missTotal = 0;
            for (uint32_t tile = 0; tile < TILES_PER_BATCH; ++tile) {
                tileHitOffsets.SetValue(tile, static_cast<int32_t>(hitTotal));
                tileMissOffsets.SetValue(tile, static_cast<int32_t>(missTotal));
                hitTotal += static_cast<uint32_t>(tileHitCounts.GetValue(tile));
                missTotal += static_cast<uint32_t>(tileMissCounts.GetValue(tile));
            }
            scalarMeta.SetValue(0, static_cast<int32_t>(hitTotal));
            scalarMeta.SetValue(8, static_cast<int32_t>(missTotal));
            scalarMeta.SetValue(16, 0);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::DataCopy(tileHitOffsetsGm[tileOffset], tileHitOffsets,
                              TILES_PER_BATCH);
            AscendC::DataCopy(tileMissOffsetsGm[tileOffset], tileMissOffsets,
                              TILES_PER_BATCH);
            AscendC::DataCopyExtParams scalarCopyParams{
                1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPad(hitCountsGm[batch], scalarMeta, scalarCopyParams);
            AscendC::DataCopyPad(missCountsGm[batch], scalarMeta[8],
                                 scalarCopyParams);
            if (hitTotal == 0) {
                AscendC::DataCopyPad(hitSparseIndicesGm[rowOffset], scalarMeta[16],
                                     scalarCopyParams);
            }
            if (missTotal == 0) {
                AscendC::DataCopyPad(missSparseIndicesGm[rowOffset], scalarMeta[16],
                                     scalarCopyParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
            if (missTotal == 0) {
                continue;
            }

            AscendC::DataCopy(hitValid, hitValidMaskGm[rowOffset], topk);
            AscendC::DataCopy(missValid, missValidMaskGm[rowOffset], topk);
            AscendC::DataCopy(selectedSlots, selectedSlotsGm[rowOffset], topk);
            AscendC::DataCopy(slotMapValues, slotMapSlotValuesGm[rowOffset], topk);
            AscendC::DataCopy(missHotDst, missHotDstIndicesGm[rowOffset], topk);
            AscendC::DataCopyExtParams rowCopyParams{
                1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int64_t> rowPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(rowMeta, deviceCacheRowIndicesGm[batch],
                                 rowCopyParams, rowPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::Duplicate(occupied, static_cast<uint32_t>(0), SLOT_BITMAP_WORDS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            for (uint32_t position = 0; position < topk; ++position) {
                if (hitValid.GetValue(position) == 0) {
                    continue;
                }
                const int32_t slot = selectedSlots.GetValue(position);
                if (slot >= 0 && static_cast<uint32_t>(slot) < topk) {
                    const uint32_t slotU32 = static_cast<uint32_t>(slot);
                    const uint32_t word = slotU32 >> 5;
                    const uint32_t bit = static_cast<uint32_t>(1U) << (slotU32 & 31U);
                    occupied.SetValue(word, occupied.GetValue(word) | bit);
                }
            }

            const int64_t cacheBase = rowMeta.GetValue(0) *
                                      static_cast<int64_t>(topk);
            uint32_t nextFreeSlot = 0;
            for (uint32_t position = 0; position < topk; ++position) {
                if (missValid.GetValue(position) == 0) {
                    continue;
                }
                while (nextFreeSlot < topk) {
                    const uint32_t word = nextFreeSlot >> 5;
                    const uint32_t occupiedWord = occupied.GetValue(word);
                    if ((nextFreeSlot & 31U) == 0 &&
                        occupiedWord == static_cast<uint32_t>(0xFFFFFFFFU)) {
                        nextFreeSlot += 32U;
                        continue;
                    }
                    const uint32_t bit = static_cast<uint32_t>(1U) <<
                                         (nextFreeSlot & 31U);
                    if ((occupiedWord & bit) == 0) {
                        break;
                    }
                    ++nextFreeSlot;
                }
                if (nextFreeSlot >= topk) {
                    continue;
                }
                missHotDst.SetValue(position,
                                    cacheBase + static_cast<int64_t>(nextFreeSlot));
                slotMapValues.SetValue(position, static_cast<int32_t>(nextFreeSlot));
                ++nextFreeSlot;
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::DataCopy(missHotDstIndicesGm[rowOffset], missHotDst, topk);
            AscendC::DataCopy(slotMapSlotValuesGm[rowOffset], slotMapValues, topk);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
    }

private:
    AscendC::GlobalTensor<int64_t> deviceCacheRowIndicesGm;
    AscendC::GlobalTensor<int32_t> hitSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> missSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> hitCountsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;
    AscendC::GlobalTensor<int64_t> missHotDstIndicesGm;
    AscendC::GlobalTensor<uint8_t> hitValidMaskGm;
    AscendC::GlobalTensor<uint8_t> missValidMaskGm;
    AscendC::GlobalTensor<int32_t> slotMapSlotValuesGm;
    AscendC::GlobalTensor<int32_t> tileHitCountsGm;
    AscendC::GlobalTensor<int32_t> tileMissCountsGm;
    AscendC::GlobalTensor<int32_t> tileHitOffsetsGm;
    AscendC::GlobalTensor<int32_t> tileMissOffsetsGm;
    AscendC::GlobalTensor<int32_t> selectedSlotsGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> occupiedBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int32Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int64Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tileMetaBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowMetaBuf;
    uint32_t batchSize = 0;
    uint32_t topk = 0;
};

extern "C" __global__ __aicore__ void sparse_kv_partition_scan(
    GM_ADDR device_cache_row_indices, GM_ADDR hit_sparse_indices, GM_ADDR miss_sparse_indices,
    GM_ADDR hit_counts, GM_ADDR miss_counts, GM_ADDR miss_hot_dst_indices,
    GM_ADDR hit_valid_mask, GM_ADDR miss_valid_mask, GM_ADDR slot_map_slot_values,
    GM_ADDR tile_hit_counts, GM_ADDR tile_miss_counts, GM_ADDR tile_hit_offsets,
    GM_ADDR tile_miss_offsets, GM_ADDR selected_slots, uint32_t batch_size, uint32_t topk)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseKvPartitionScan kernel;
    kernel.Init(device_cache_row_indices, hit_sparse_indices, miss_sparse_indices,
                hit_counts, miss_counts, miss_hot_dst_indices, hit_valid_mask,
                miss_valid_mask, slot_map_slot_values, tile_hit_counts,
                tile_miss_counts, tile_hit_offsets, tile_miss_offsets,
                selected_slots, batch_size, topk, &pipe);
    kernel.Process();
}
