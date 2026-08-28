/**
 * @file sparse_kv_partition_scatter_kernel.cpp
 * @brief Tile-parallel stable scatter and miss-slot assignment.
 */

#include "kernel_operator.h"

constexpr uint32_t FIXED_TOPK = 2048;
constexpr uint32_t TOPK_TILE_LEN = 64;
constexpr uint32_t TILES_PER_BATCH = FIXED_TOPK / TOPK_TILE_LEN;
constexpr uint32_t SLOT_BITMAP_WORDS = FIXED_TOPK / 32;

class KernelSparseKvPartitionScatter
{
public:
    __aicore__ inline KernelSparseKvPartitionScatter() {}

    __aicore__ inline void Init(
        GM_ADDR deviceCacheRowIndices, GM_ADDR tileHitCounts,
        GM_ADDR tileMissCounts, GM_ADDR tileHitOffsets,
        GM_ADDR tileMissOffsets, GM_ADDR selectedSlots,
        GM_ADDR occupiedBitmaps, GM_ADDR freeSlotPrefixes,
        GM_ADDR hitSparseIndices, GM_ADDR missSparseIndices,
        GM_ADDR missHotDstIndices, GM_ADDR slotMapSlotValues,
        uint32_t batchSize, uint32_t topk, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->topk = topk;
        const uint64_t planSize = static_cast<uint64_t>(batchSize) * topk;
        const uint64_t tileSize =
            static_cast<uint64_t>(batchSize) * TILES_PER_BATCH;
        deviceCacheRowIndicesGm.SetGlobalBuffer(
            (__gm__ int64_t *)deviceCacheRowIndices, batchSize);
        tileHitCountsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileHitCounts, tileSize);
        tileMissCountsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileMissCounts, tileSize);
        tileHitOffsetsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileHitOffsets, tileSize);
        tileMissOffsetsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileMissOffsets, tileSize);
        selectedSlotsGm.SetGlobalBuffer(
            (__gm__ int32_t *)selectedSlots, planSize);
        occupiedBitmapsGm.SetGlobalBuffer(
            (__gm__ uint32_t *)occupiedBitmaps,
            static_cast<uint64_t>(batchSize) * SLOT_BITMAP_WORDS);
        freeSlotPrefixesGm.SetGlobalBuffer(
            (__gm__ int32_t *)freeSlotPrefixes,
            static_cast<uint64_t>(batchSize) * SLOT_BITMAP_WORDS);
        hitSparseIndicesGm.SetGlobalBuffer(
            (__gm__ int32_t *)hitSparseIndices, planSize);
        missSparseIndicesGm.SetGlobalBuffer(
            (__gm__ int32_t *)missSparseIndices, planSize);
        missHotDstIndicesGm.SetGlobalBuffer(
            (__gm__ int64_t *)missHotDstIndices, planSize);
        slotMapSlotValuesGm.SetGlobalBuffer(
            (__gm__ int32_t *)slotMapSlotValues, planSize);

        pipe->InitBuffer(
            int32Buf, 4U * TOPK_TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(
            int64Buf, TOPK_TILE_LEN * sizeof(int64_t));
        pipe->InitBuffer(
            bitmapBuf, 2U * SLOT_BITMAP_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(metaBuf, 32U * sizeof(int32_t));
        pipe->InitBuffer(rowMetaBuf, 4U * sizeof(int64_t));
    }

    __aicore__ inline uint32_t CountTrailingZeros(uint32_t value) const
    {
        uint32_t count = 0;
        while ((value & 1U) == 0U) {
            value >>= 1U;
            ++count;
        }
        return count;
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        const uint32_t taskCount = batchSize * TILES_PER_BATCH;
        AscendC::LocalTensor<int32_t> int32Base = int32Buf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> hitLocal = int32Base;
        AscendC::LocalTensor<int32_t> missLocal =
            int32Base[TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> selectedSlots =
            int32Base[2U * TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> slotMapValues =
            int32Base[3U * TOPK_TILE_LEN];
        AscendC::LocalTensor<int64_t> missHotDst =
            int64Buf.Get<int64_t>();
        AscendC::LocalTensor<uint32_t> bitmapBase =
            bitmapBuf.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> occupied = bitmapBase;
        AscendC::LocalTensor<int32_t> freePrefixes =
            bitmapBase[SLOT_BITMAP_WORDS].ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> meta = metaBuf.Get<int32_t>();
        AscendC::LocalTensor<int64_t> rowMeta = rowMetaBuf.Get<int64_t>();

        for (uint32_t task = workerIndex; task < taskCount;
             task += workerCount) {
            const uint32_t batch = task / TILES_PER_BATCH;
            const uint32_t tile = task % TILES_PER_BATCH;
            const uint32_t tileBegin = tile * TOPK_TILE_LEN;
            const uint32_t rowOffset = batch * topk;
            const uint32_t gmOffset = rowOffset + tileBegin;
            const uint32_t bitmapOffset = batch * SLOT_BITMAP_WORDS;

            AscendC::DataCopy(selectedSlots,
                              selectedSlotsGm[gmOffset], TOPK_TILE_LEN);
            AscendC::DataCopy(occupied,
                              occupiedBitmapsGm[bitmapOffset],
                              SLOT_BITMAP_WORDS);
            AscendC::DataCopy(freePrefixes,
                              freeSlotPrefixesGm[bitmapOffset],
                              SLOT_BITMAP_WORDS);
            AscendC::DataCopyExtParams scalarCopyParams{
                1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> scalarPadParams{
                false, 0, 0, 0};
            AscendC::DataCopyPad(meta, tileHitCountsGm[task],
                                 scalarCopyParams, scalarPadParams);
            AscendC::DataCopyPad(meta[8], tileMissCountsGm[task],
                                 scalarCopyParams, scalarPadParams);
            AscendC::DataCopyPad(meta[16], tileHitOffsetsGm[task],
                                 scalarCopyParams, scalarPadParams);
            AscendC::DataCopyPad(meta[24], tileMissOffsetsGm[task],
                                 scalarCopyParams, scalarPadParams);
            AscendC::DataCopyExtParams rowCopyParams{
                1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int64_t> rowPadParams{
                false, 0, 0, 0};
            AscendC::DataCopyPad(rowMeta,
                                 deviceCacheRowIndicesGm[batch],
                                 rowCopyParams, rowPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

            const uint32_t expectedHit =
                static_cast<uint32_t>(meta.GetValue(0));
            const uint32_t expectedMiss =
                static_cast<uint32_t>(meta.GetValue(8));
            const uint32_t hitOffset =
                static_cast<uint32_t>(meta.GetValue(16));
            const uint32_t missOffset =
                static_cast<uint32_t>(meta.GetValue(24));
            const int64_t cacheBase =
                rowMeta.GetValue(0) * static_cast<int64_t>(topk);

            uint32_t freeWord = 0;
            uint32_t freeBits = 0;
            if (expectedMiss > 0) {
                while (freeWord + 1U < SLOT_BITMAP_WORDS &&
                       missOffset >= static_cast<uint32_t>(
                           freePrefixes.GetValue(freeWord + 1U))) {
                    ++freeWord;
                }
                freeBits = ~occupied.GetValue(freeWord);
                uint32_t skip = missOffset - static_cast<uint32_t>(
                    freePrefixes.GetValue(freeWord));
                while (skip > 0 && freeBits != 0U) {
                    freeBits &= freeBits - 1U;
                    --skip;
                }
            }

            uint32_t localHit = 0;
            uint32_t localMiss = 0;
            for (uint32_t i = 0; i < TOPK_TILE_LEN; ++i) {
                const int32_t selectedSlot = selectedSlots.GetValue(i);
                const bool hit = selectedSlot >= 0;
                const bool miss = selectedSlot == -2;
                missHotDst.SetValue(i, cacheBase);
                slotMapValues.SetValue(i, hit ? selectedSlot : -1);

                if (hit) {
                    hitLocal.SetValue(localHit++,
                                      static_cast<int32_t>(tileBegin + i));
                }
                if (!miss) {
                    continue;
                }
                missLocal.SetValue(localMiss++,
                                   static_cast<int32_t>(tileBegin + i));
                while (freeBits == 0U &&
                       freeWord + 1U < SLOT_BITMAP_WORDS) {
                    ++freeWord;
                    freeBits = ~occupied.GetValue(freeWord);
                }
                if (freeBits == 0U) {
                    continue;
                }
                const uint32_t bit = CountTrailingZeros(freeBits);
                freeBits &= freeBits - 1U;
                const uint32_t assignedSlot = freeWord * 32U + bit;
                missHotDst.SetValue(
                    i, cacheBase + static_cast<int64_t>(assignedSlot));
                slotMapValues.SetValue(
                    i, static_cast<int32_t>(assignedSlot));
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            if (expectedHit > 0) {
                AscendC::DataCopyExtParams copyParams{
                    1,
                    expectedHit * static_cast<uint32_t>(sizeof(int32_t)),
                    0, 0, 0};
                AscendC::DataCopyPad(
                    hitSparseIndicesGm[rowOffset + hitOffset],
                    hitLocal, copyParams);
            }
            if (expectedMiss > 0) {
                AscendC::DataCopyExtParams copyParams{
                    1,
                    expectedMiss * static_cast<uint32_t>(sizeof(int32_t)),
                    0, 0, 0};
                AscendC::DataCopyPad(
                    missSparseIndicesGm[rowOffset + missOffset],
                    missLocal, copyParams);
            }
            AscendC::DataCopy(
                missHotDstIndicesGm[gmOffset], missHotDst,
                TOPK_TILE_LEN);
            AscendC::DataCopy(
                slotMapSlotValuesGm[gmOffset], slotMapValues,
                TOPK_TILE_LEN);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
    }

private:
    AscendC::GlobalTensor<int64_t> deviceCacheRowIndicesGm;
    AscendC::GlobalTensor<int32_t> tileHitCountsGm;
    AscendC::GlobalTensor<int32_t> tileMissCountsGm;
    AscendC::GlobalTensor<int32_t> tileHitOffsetsGm;
    AscendC::GlobalTensor<int32_t> tileMissOffsetsGm;
    AscendC::GlobalTensor<int32_t> selectedSlotsGm;
    AscendC::GlobalTensor<uint32_t> occupiedBitmapsGm;
    AscendC::GlobalTensor<int32_t> freeSlotPrefixesGm;
    AscendC::GlobalTensor<int32_t> hitSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> missSparseIndicesGm;
    AscendC::GlobalTensor<int64_t> missHotDstIndicesGm;
    AscendC::GlobalTensor<int32_t> slotMapSlotValuesGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int32Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int64Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bitmapBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> metaBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowMetaBuf;
    uint32_t batchSize = 0;
    uint32_t topk = 0;
};

extern "C" __global__ __aicore__ void sparse_kv_partition_scatter(
    GM_ADDR device_cache_row_indices, GM_ADDR tile_hit_counts,
    GM_ADDR tile_miss_counts, GM_ADDR tile_hit_offsets,
    GM_ADDR tile_miss_offsets, GM_ADDR selected_slots,
    GM_ADDR occupied_bitmaps, GM_ADDR free_slot_prefixes,
    GM_ADDR hit_sparse_indices, GM_ADDR miss_sparse_indices,
    GM_ADDR miss_hot_dst_indices, GM_ADDR slot_map_slot_values,
    uint32_t batch_size, uint32_t topk)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseKvPartitionScatter kernel;
    kernel.Init(device_cache_row_indices, tile_hit_counts,
                tile_miss_counts, tile_hit_offsets,
                tile_miss_offsets, selected_slots, occupied_bitmaps,
                free_slot_prefixes, hit_sparse_indices,
                miss_sparse_indices, miss_hot_dst_indices,
                slot_map_slot_values, batch_size, topk, &pipe);
    kernel.Process();
}
