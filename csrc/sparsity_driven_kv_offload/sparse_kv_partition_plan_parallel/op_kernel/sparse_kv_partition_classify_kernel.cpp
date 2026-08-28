/**
 * @file sparse_kv_partition_classify_kernel.cpp
 * @brief Tile-parallel hit/miss classification for sparse KV planning.
 */

#include "kernel_operator.h"

constexpr uint32_t FIXED_TOPK = 2048;
constexpr uint32_t TOPK_TILE_LEN = 64;
constexpr uint32_t TILES_PER_BATCH = FIXED_TOPK / TOPK_TILE_LEN;
constexpr uint32_t SLOT_BITMAP_WORDS = FIXED_TOPK / 32;

class KernelSparseKvPartitionClassify
{
public:
    __aicore__ inline KernelSparseKvPartitionClassify() {}

    __aicore__ inline void Init(
        GM_ADDR tokenOnDevice, GM_ADDR deviceTokenPos, GM_ADDR topkIndices,
        GM_ADDR deviceCacheRowIndices, GM_ADDR slotMapRowIndices, GM_ADDR validTopkMask,
        GM_ADDR hitSparseIndices, GM_ADDR missSparseIndices,
        GM_ADDR hitSrcIndices, GM_ADDR missSrcIndices,
        GM_ADDR hitValidMask, GM_ADDR missValidMask, GM_ADDR slotMapFlatIndices,
        GM_ADDR tileHitCounts, GM_ADDR tileMissCounts, GM_ADDR selectedSlots,
        GM_ADDR tileOccupiedBitmaps, uint32_t batchSize, uint32_t topk,
        uint32_t maxContextLen, uint32_t slotMapWidth, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->topk = topk;
        this->maxContextLen = maxContextLen;
        this->slotMapWidth = slotMapWidth;

        const uint64_t planSize = static_cast<uint64_t>(batchSize) * topk;
        const uint64_t tileSize = static_cast<uint64_t>(batchSize) * TILES_PER_BATCH;
        tokenOnDeviceGm.SetGlobalBuffer((__gm__ int32_t *)tokenOnDevice, planSize);
        deviceTokenPosGm.SetGlobalBuffer((__gm__ int32_t *)deviceTokenPos, planSize);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices, planSize);
        deviceCacheRowIndicesGm.SetGlobalBuffer((__gm__ int64_t *)deviceCacheRowIndices, batchSize);
        slotMapRowIndicesGm.SetGlobalBuffer((__gm__ int64_t *)slotMapRowIndices, batchSize);
        validTopkMaskGm.SetGlobalBuffer((__gm__ uint8_t *)validTopkMask, planSize);

        hitSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)hitSparseIndices, planSize);
        missSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)missSparseIndices, planSize);
        hitSrcIndicesGm.SetGlobalBuffer((__gm__ int64_t *)hitSrcIndices, planSize);
        missSrcIndicesGm.SetGlobalBuffer((__gm__ int64_t *)missSrcIndices, planSize);
        hitValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)hitValidMask, planSize);
        missValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)missValidMask, planSize);
        slotMapFlatIndicesGm.SetGlobalBuffer((__gm__ int64_t *)slotMapFlatIndices, planSize);
        tileHitCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileHitCounts, tileSize);
        tileMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileMissCounts, tileSize);
        selectedSlotsGm.SetGlobalBuffer((__gm__ int32_t *)selectedSlots, planSize);
        tileOccupiedBitmapsGm.SetGlobalBuffer(
            (__gm__ uint32_t *)tileOccupiedBitmaps,
            tileSize * SLOT_BITMAP_WORDS);

        pipe->InitBuffer(int32Buf, 6U * TOPK_TILE_LEN * sizeof(int32_t));
        pipe->InitBuffer(maskBuf, 3U * TOPK_TILE_LEN * sizeof(uint8_t));
        pipe->InitBuffer(int64Buf, 3U * TOPK_TILE_LEN * sizeof(int64_t));
        pipe->InitBuffer(bitmapBuf, SLOT_BITMAP_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(rowMetaBuf, 8U * sizeof(int64_t));
        pipe->InitBuffer(countBuf, 16U * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        const uint32_t taskCount = batchSize * TILES_PER_BATCH;

        AscendC::LocalTensor<int32_t> int32Base = int32Buf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> tokenOnDevice = int32Base;
        AscendC::LocalTensor<int32_t> deviceTokenPos = int32Base[TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> topkIndices = int32Base[2U * TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> hitSparseClear = int32Base[3U * TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> missSparseClear = int32Base[4U * TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> selectedSlots = int32Base[5U * TOPK_TILE_LEN];

        AscendC::LocalTensor<uint8_t> maskBase = maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> validTopk = maskBase;
        AscendC::LocalTensor<uint8_t> hitValid = maskBase[TOPK_TILE_LEN];
        AscendC::LocalTensor<uint8_t> missValid = maskBase[2U * TOPK_TILE_LEN];

        AscendC::LocalTensor<int64_t> int64Base = int64Buf.Get<int64_t>();
        AscendC::LocalTensor<int64_t> hitSrc = int64Base;
        AscendC::LocalTensor<int64_t> missSrc = int64Base[TOPK_TILE_LEN];
        AscendC::LocalTensor<int64_t> slotMapFlat = int64Base[2U * TOPK_TILE_LEN];
        AscendC::LocalTensor<uint32_t> occupied = bitmapBuf.Get<uint32_t>();
        AscendC::LocalTensor<int64_t> rowMeta = rowMetaBuf.Get<int64_t>();
        AscendC::LocalTensor<int32_t> counts = countBuf.Get<int32_t>();

        for (uint32_t task = workerIndex; task < taskCount; task += workerCount) {
            const uint32_t batch = task / TILES_PER_BATCH;
            const uint32_t tile = task % TILES_PER_BATCH;
            const uint32_t tileOffset = tile * TOPK_TILE_LEN;
            const uint32_t rowOffset = batch * topk;
            const uint32_t gmOffset = rowOffset + tileOffset;

            AscendC::DataCopy(tokenOnDevice, tokenOnDeviceGm[gmOffset], TOPK_TILE_LEN);
            AscendC::DataCopy(deviceTokenPos, deviceTokenPosGm[gmOffset], TOPK_TILE_LEN);
            AscendC::DataCopy(topkIndices, topkIndicesGm[gmOffset], TOPK_TILE_LEN);
            AscendC::DataCopy(validTopk, validTopkMaskGm[gmOffset], TOPK_TILE_LEN);
            AscendC::DataCopyExtParams rowCopyParams{
                1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int64_t> rowPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(rowMeta, deviceCacheRowIndicesGm[batch],
                                 rowCopyParams, rowPadParams);
            AscendC::DataCopyPad(rowMeta[4], slotMapRowIndicesGm[batch],
                                 rowCopyParams, rowPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

            const int64_t cacheRow = rowMeta.GetValue(0);
            const int64_t slotMapRow = rowMeta.GetValue(4);
            const int64_t cacheBase = cacheRow * static_cast<int64_t>(topk);
            const int64_t hostBase = cacheRow * static_cast<int64_t>(maxContextLen);
            const int64_t slotMapBase = slotMapRow * static_cast<int64_t>(slotMapWidth);
            uint32_t hitCount = 0;
            uint32_t missCount = 0;
            AscendC::Duplicate(occupied, static_cast<uint32_t>(0),
                               SLOT_BITMAP_WORDS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

            for (uint32_t i = 0; i < TOPK_TILE_LEN; ++i) {
                const bool valid = validTopk.GetValue(i) != 0;
                const int32_t logicalToken = topkIndices.GetValue(i);
                const int32_t rawSlot = deviceTokenPos.GetValue(i);
                const bool hit = valid && tokenOnDevice.GetValue(i) != 0 && rawSlot >= 0 &&
                                 static_cast<uint32_t>(rawSlot) < topk;
                const bool miss = valid && !hit;
                hitCount += hit ? 1U : 0U;
                missCount += miss ? 1U : 0U;

                hitSparseClear.SetValue(i, -1);
                missSparseClear.SetValue(i, -1);
                hitValid.SetValue(i, hit ? 1U : 0U);
                missValid.SetValue(i, miss ? 1U : 0U);

                int32_t safeToken = logicalToken;
                if (safeToken < 0) {
                    safeToken = 0;
                } else if (static_cast<uint32_t>(safeToken) >= maxContextLen) {
                    safeToken = static_cast<int32_t>(maxContextLen - 1U);
                }
                const int32_t safeSlot = hit ? rawSlot : 0;
                hitSrc.SetValue(i, cacheBase + static_cast<int64_t>(safeSlot));
                missSrc.SetValue(i, hostBase + static_cast<int64_t>(safeToken));
                selectedSlots.SetValue(i, hit ? rawSlot : (miss ? -2 : -1));
                if (hit) {
                    const uint32_t slotU32 = static_cast<uint32_t>(rawSlot);
                    const uint32_t word = slotU32 >> 5;
                    const uint32_t bit = static_cast<uint32_t>(1U)
                                         << (slotU32 & 31U);
                    occupied.SetValue(word, occupied.GetValue(word) | bit);
                }

                const int64_t publishToken = valid ? static_cast<int64_t>(logicalToken)
                                                   : static_cast<int64_t>(maxContextLen);
                slotMapFlat.SetValue(i, slotMapBase + publishToken);
            }

            counts.SetValue(0, static_cast<int32_t>(hitCount));
            counts.SetValue(8, static_cast<int32_t>(missCount));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::DataCopyExtParams countCopyParams{
                1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPad(tileHitCountsGm[task], counts, countCopyParams);
            AscendC::DataCopyPad(tileMissCountsGm[task], counts[8], countCopyParams);
            AscendC::DataCopy(hitSparseIndicesGm[gmOffset], hitSparseClear, TOPK_TILE_LEN);
            AscendC::DataCopy(missSparseIndicesGm[gmOffset], missSparseClear, TOPK_TILE_LEN);
            AscendC::DataCopy(hitSrcIndicesGm[gmOffset], hitSrc, TOPK_TILE_LEN);
            AscendC::DataCopy(missSrcIndicesGm[gmOffset], missSrc, TOPK_TILE_LEN);
            AscendC::DataCopy(hitValidMaskGm[gmOffset], hitValid, TOPK_TILE_LEN);
            AscendC::DataCopy(missValidMaskGm[gmOffset], missValid, TOPK_TILE_LEN);
            AscendC::DataCopy(slotMapFlatIndicesGm[gmOffset], slotMapFlat, TOPK_TILE_LEN);
            AscendC::DataCopy(selectedSlotsGm[gmOffset], selectedSlots, TOPK_TILE_LEN);
            AscendC::DataCopy(
                tileOccupiedBitmapsGm[task * SLOT_BITMAP_WORDS], occupied,
                SLOT_BITMAP_WORDS);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
    }

private:
    AscendC::GlobalTensor<int32_t> tokenOnDeviceGm;
    AscendC::GlobalTensor<int32_t> deviceTokenPosGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int64_t> deviceCacheRowIndicesGm;
    AscendC::GlobalTensor<int64_t> slotMapRowIndicesGm;
    AscendC::GlobalTensor<uint8_t> validTopkMaskGm;
    AscendC::GlobalTensor<int32_t> hitSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> missSparseIndicesGm;
    AscendC::GlobalTensor<int64_t> hitSrcIndicesGm;
    AscendC::GlobalTensor<int64_t> missSrcIndicesGm;
    AscendC::GlobalTensor<uint8_t> hitValidMaskGm;
    AscendC::GlobalTensor<uint8_t> missValidMaskGm;
    AscendC::GlobalTensor<int64_t> slotMapFlatIndicesGm;
    AscendC::GlobalTensor<int32_t> tileHitCountsGm;
    AscendC::GlobalTensor<int32_t> tileMissCountsGm;
    AscendC::GlobalTensor<int32_t> selectedSlotsGm;
    AscendC::GlobalTensor<uint32_t> tileOccupiedBitmapsGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int32Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> int64Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bitmapBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rowMetaBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> countBuf;
    uint32_t batchSize = 0;
    uint32_t topk = 0;
    uint32_t maxContextLen = 0;
    uint32_t slotMapWidth = 0;
};

extern "C" __global__ __aicore__ void sparse_kv_partition_classify(
    GM_ADDR token_on_device, GM_ADDR device_token_pos, GM_ADDR topk_indices,
    GM_ADDR device_cache_row_indices, GM_ADDR slot_map_row_indices, GM_ADDR valid_topk_mask,
    GM_ADDR hit_sparse_indices, GM_ADDR miss_sparse_indices,
    GM_ADDR hit_src_indices, GM_ADDR miss_src_indices,
    GM_ADDR hit_valid_mask, GM_ADDR miss_valid_mask, GM_ADDR slot_map_flat_indices,
    GM_ADDR tile_hit_counts, GM_ADDR tile_miss_counts, GM_ADDR selected_slots,
    GM_ADDR tile_occupied_bitmaps, uint32_t batch_size, uint32_t topk,
    uint32_t max_context_len, uint32_t slot_map_width)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseKvPartitionClassify kernel;
    kernel.Init(token_on_device, device_token_pos, topk_indices, device_cache_row_indices,
                slot_map_row_indices, valid_topk_mask, hit_sparse_indices, miss_sparse_indices,
                hit_src_indices, miss_src_indices, hit_valid_mask, miss_valid_mask,
                slot_map_flat_indices, tile_hit_counts, tile_miss_counts,
                selected_slots, tile_occupied_bitmaps, batch_size, topk,
                max_context_len, slot_map_width, &pipe);
    kernel.Process();
}
