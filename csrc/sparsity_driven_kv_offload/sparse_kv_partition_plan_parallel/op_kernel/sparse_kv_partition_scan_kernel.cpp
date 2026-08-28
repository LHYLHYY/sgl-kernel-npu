/**
 * @file sparse_kv_partition_scan_kernel.cpp
 * @brief Request-level tile scan and resident-slot bitmap reduction.
 */

#include "kernel_operator.h"

constexpr uint32_t FIXED_TOPK = 2048;
constexpr uint32_t TILES_PER_BATCH = 32;
constexpr uint32_t SLOT_BITMAP_WORDS = FIXED_TOPK / 32;
constexpr uint32_t TILE_BITMAP_ELEMENTS =
    TILES_PER_BATCH * SLOT_BITMAP_WORDS;

class KernelSparseKvPartitionScan
{
public:
    __aicore__ inline KernelSparseKvPartitionScan() {}

    __aicore__ inline void Init(
        GM_ADDR hitSparseIndices, GM_ADDR missSparseIndices,
        GM_ADDR hitCounts, GM_ADDR missCounts, GM_ADDR tileHitCounts,
        GM_ADDR tileMissCounts, GM_ADDR tileHitOffsets,
        GM_ADDR tileMissOffsets, GM_ADDR tileOccupiedBitmaps,
        GM_ADDR occupiedBitmaps, GM_ADDR freeSlotPrefixes,
        uint32_t batchSize, uint32_t topk, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->topk = topk;
        const uint64_t planSize = static_cast<uint64_t>(batchSize) * topk;
        const uint64_t tileSize =
            static_cast<uint64_t>(batchSize) * TILES_PER_BATCH;
        hitSparseIndicesGm.SetGlobalBuffer(
            (__gm__ int32_t *)hitSparseIndices, planSize);
        missSparseIndicesGm.SetGlobalBuffer(
            (__gm__ int32_t *)missSparseIndices, planSize);
        hitCountsGm.SetGlobalBuffer((__gm__ int32_t *)hitCounts, batchSize);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
        tileHitCountsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileHitCounts, tileSize);
        tileMissCountsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileMissCounts, tileSize);
        tileHitOffsetsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileHitOffsets, tileSize);
        tileMissOffsetsGm.SetGlobalBuffer(
            (__gm__ int32_t *)tileMissOffsets, tileSize);
        tileOccupiedBitmapsGm.SetGlobalBuffer(
            (__gm__ uint32_t *)tileOccupiedBitmaps,
            static_cast<uint64_t>(batchSize) * TILE_BITMAP_ELEMENTS);
        occupiedBitmapsGm.SetGlobalBuffer(
            (__gm__ uint32_t *)occupiedBitmaps,
            static_cast<uint64_t>(batchSize) * SLOT_BITMAP_WORDS);
        freeSlotPrefixesGm.SetGlobalBuffer(
            (__gm__ int32_t *)freeSlotPrefixes,
            static_cast<uint64_t>(batchSize) * SLOT_BITMAP_WORDS);

        pipe->InitBuffer(
            tileBitmapBuf, TILE_BITMAP_ELEMENTS * sizeof(uint32_t));
        pipe->InitBuffer(reduceBuf, 2U * SLOT_BITMAP_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(prefixBuf, SLOT_BITMAP_WORDS * sizeof(int32_t));
        // Four 32-entry tile rows plus scalar staging. All metadata crosses
        // kernel boundaries through MTE, not GlobalTensor scalar DCache.
        pipe->InitBuffer(
            tileMetaBuf, 5U * TILES_PER_BATCH * sizeof(int32_t));
    }

    __aicore__ inline uint32_t PopCount32(uint32_t value) const
    {
        // Fixed-cost SWAR popcount avoids depending on the CANN 9.1-only
        // scalar GetBitCount API.
        value = value - ((value >> 1U) & 0x55555555U);
        value = (value & 0x33333333U) +
                ((value >> 2U) & 0x33333333U);
        value = (value + (value >> 4U)) & 0x0F0F0F0FU;
        value = value + (value >> 8U);
        value = value + (value >> 16U);
        return value & 0x3FU;
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        AscendC::LocalTensor<uint32_t> tileBitmaps =
            tileBitmapBuf.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> reduceBase = reduceBuf.Get<uint32_t>();
        AscendC::LocalTensor<uint32_t> reduceA = reduceBase;
        AscendC::LocalTensor<uint32_t> reduceB =
            reduceBase[SLOT_BITMAP_WORDS];
        AscendC::LocalTensor<int32_t> freePrefixes =
            prefixBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> tileMeta =
            tileMetaBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> tileHitCounts = tileMeta;
        AscendC::LocalTensor<int32_t> tileMissCounts =
            tileMeta[TILES_PER_BATCH];
        AscendC::LocalTensor<int32_t> tileHitOffsets =
            tileMeta[2U * TILES_PER_BATCH];
        AscendC::LocalTensor<int32_t> tileMissOffsets =
            tileMeta[3U * TILES_PER_BATCH];
        AscendC::LocalTensor<int32_t> scalarMeta =
            tileMeta[4U * TILES_PER_BATCH];

        for (uint32_t batch = workerIndex; batch < batchSize;
             batch += workerCount) {
            const uint32_t rowOffset = batch * topk;
            const uint32_t tileOffset = batch * TILES_PER_BATCH;
            const uint32_t bitmapOffset = batch * TILE_BITMAP_ELEMENTS;
            const uint32_t requestBitmapOffset =
                batch * SLOT_BITMAP_WORDS;

            AscendC::DataCopy(tileHitCounts,
                              tileHitCountsGm[tileOffset],
                              TILES_PER_BATCH);
            AscendC::DataCopy(tileMissCounts,
                              tileMissCountsGm[tileOffset],
                              TILES_PER_BATCH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::DataCopy(tileBitmaps,
                              tileOccupiedBitmapsGm[bitmapOffset],
                              TILE_BITMAP_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);

            uint32_t hitTotal = 0;
            uint32_t missTotal = 0;
            for (uint32_t tile = 0; tile < TILES_PER_BATCH; ++tile) {
                tileHitOffsets.SetValue(tile,
                                        static_cast<int32_t>(hitTotal));
                tileMissOffsets.SetValue(tile,
                                         static_cast<int32_t>(missTotal));
                hitTotal += static_cast<uint32_t>(
                    tileHitCounts.GetValue(tile));
                missTotal += static_cast<uint32_t>(
                    tileMissCounts.GetValue(tile));
            }
            scalarMeta.SetValue(0, static_cast<int32_t>(hitTotal));
            scalarMeta.SetValue(8, static_cast<int32_t>(missTotal));
            scalarMeta.SetValue(16, 0);

            // OR-reduce the 32 tile-local 2048-bit slot maps. Reinterpreting
            // uint32 as uint16 follows the cross-version Ascend C contract for
            // bitwise Or on A2/A3 products.
            AscendC::LocalTensor<uint16_t> reduceA16 =
                reduceA.ReinterpretCast<uint16_t>();
            AscendC::LocalTensor<uint16_t> reduceB16 =
                reduceB.ReinterpretCast<uint16_t>();
            AscendC::LocalTensor<uint16_t> tileBitmaps16 =
                tileBitmaps.ReinterpretCast<uint16_t>();
            constexpr int32_t bitmapHalfWords =
                static_cast<int32_t>(2U * SLOT_BITMAP_WORDS);
            constexpr uint32_t tileHalfWords = 2U * SLOT_BITMAP_WORDS;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Or(reduceA16, tileBitmaps16, tileBitmaps16,
                        bitmapHalfWords);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t tile = 1; tile < TILES_PER_BATCH; tile += 2U) {
                AscendC::Or(
                    reduceB16, reduceA16,
                    tileBitmaps16[tile * tileHalfWords],
                    bitmapHalfWords);
                AscendC::PipeBarrier<PIPE_V>();
                if (tile + 1U < TILES_PER_BATCH) {
                    AscendC::Or(
                        reduceA16, reduceB16,
                        tileBitmaps16[(tile + 1U) * tileHalfWords],
                        bitmapHalfWords);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
            // There are 31 reductions after tile zero, so reduceB is final.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            uint32_t freeTotal = 0;
            for (uint32_t word = 0; word < SLOT_BITMAP_WORDS; ++word) {
                freePrefixes.SetValue(word,
                                      static_cast<int32_t>(freeTotal));
                freeTotal += 32U - PopCount32(reduceB.GetValue(word));
            }

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::DataCopy(tileHitOffsetsGm[tileOffset],
                              tileHitOffsets, TILES_PER_BATCH);
            AscendC::DataCopy(tileMissOffsetsGm[tileOffset],
                              tileMissOffsets, TILES_PER_BATCH);
            AscendC::DataCopy(
                occupiedBitmapsGm[requestBitmapOffset], reduceB,
                SLOT_BITMAP_WORDS);
            AscendC::DataCopy(
                freeSlotPrefixesGm[requestBitmapOffset], freePrefixes,
                SLOT_BITMAP_WORDS);
            AscendC::DataCopyExtParams scalarCopyParams{
                1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPad(hitCountsGm[batch], scalarMeta,
                                 scalarCopyParams);
            AscendC::DataCopyPad(missCountsGm[batch], scalarMeta[8],
                                 scalarCopyParams);
            if (hitTotal == 0) {
                AscendC::DataCopyPad(hitSparseIndicesGm[rowOffset],
                                     scalarMeta[16], scalarCopyParams);
            }
            if (missTotal == 0) {
                AscendC::DataCopyPad(missSparseIndicesGm[rowOffset],
                                     scalarMeta[16], scalarCopyParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
    }

private:
    AscendC::GlobalTensor<int32_t> hitSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> missSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> hitCountsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;
    AscendC::GlobalTensor<int32_t> tileHitCountsGm;
    AscendC::GlobalTensor<int32_t> tileMissCountsGm;
    AscendC::GlobalTensor<int32_t> tileHitOffsetsGm;
    AscendC::GlobalTensor<int32_t> tileMissOffsetsGm;
    AscendC::GlobalTensor<uint32_t> tileOccupiedBitmapsGm;
    AscendC::GlobalTensor<uint32_t> occupiedBitmapsGm;
    AscendC::GlobalTensor<int32_t> freeSlotPrefixesGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tileBitmapBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prefixBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tileMetaBuf;
    uint32_t batchSize = 0;
    uint32_t topk = 0;
};

extern "C" __global__ __aicore__ void sparse_kv_partition_scan(
    GM_ADDR hit_sparse_indices, GM_ADDR miss_sparse_indices,
    GM_ADDR hit_counts, GM_ADDR miss_counts, GM_ADDR tile_hit_counts,
    GM_ADDR tile_miss_counts, GM_ADDR tile_hit_offsets,
    GM_ADDR tile_miss_offsets, GM_ADDR tile_occupied_bitmaps,
    GM_ADDR occupied_bitmaps, GM_ADDR free_slot_prefixes,
    uint32_t batch_size, uint32_t topk)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseKvPartitionScan kernel;
    kernel.Init(hit_sparse_indices, miss_sparse_indices, hit_counts,
                miss_counts, tile_hit_counts, tile_miss_counts,
                tile_hit_offsets, tile_miss_offsets,
                tile_occupied_bitmaps, occupied_bitmaps,
                free_slot_prefixes, batch_size, topk, &pipe);
    kernel.Process();
}
