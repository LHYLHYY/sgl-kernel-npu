/**
 * @file sparse_kv_partition_scatter_kernel.cpp
 * @brief Tile-parallel stable scatter into compact hit/miss index prefixes.
 */

#include "kernel_operator.h"

constexpr uint32_t FIXED_TOPK = 2048;
constexpr uint32_t TOPK_TILE_LEN = 64;
constexpr uint32_t TILES_PER_BATCH = FIXED_TOPK / TOPK_TILE_LEN;

class KernelSparseKvPartitionScatter
{
public:
    __aicore__ inline KernelSparseKvPartitionScatter() {}

    __aicore__ inline void Init(
        GM_ADDR hitValidMask, GM_ADDR missValidMask, GM_ADDR tileHitCounts,
        GM_ADDR tileMissCounts, GM_ADDR tileHitOffsets, GM_ADDR tileMissOffsets,
        GM_ADDR hitSparseIndices, GM_ADDR missSparseIndices, uint32_t batchSize,
        uint32_t topk, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->topk = topk;
        const uint64_t planSize = static_cast<uint64_t>(batchSize) * topk;
        const uint64_t tileSize = static_cast<uint64_t>(batchSize) * TILES_PER_BATCH;
        hitValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)hitValidMask, planSize);
        missValidMaskGm.SetGlobalBuffer((__gm__ uint8_t *)missValidMask, planSize);
        tileHitCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileHitCounts, tileSize);
        tileMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)tileMissCounts, tileSize);
        tileHitOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)tileHitOffsets, tileSize);
        tileMissOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)tileMissOffsets, tileSize);
        hitSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)hitSparseIndices, planSize);
        missSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)missSparseIndices, planSize);
        pipe->InitBuffer(maskBuf, 2U * TOPK_TILE_LEN * sizeof(uint8_t));
        pipe->InitBuffer(indexBuf, 2U * TOPK_TILE_LEN * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        const uint32_t taskCount = batchSize * TILES_PER_BATCH;
        AscendC::LocalTensor<uint8_t> maskBase = maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> hitValid = maskBase;
        AscendC::LocalTensor<uint8_t> missValid = maskBase[TOPK_TILE_LEN];
        AscendC::LocalTensor<int32_t> indexBase = indexBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> hitLocal = indexBase;
        AscendC::LocalTensor<int32_t> missLocal = indexBase[TOPK_TILE_LEN];

        for (uint32_t task = workerIndex; task < taskCount; task += workerCount) {
            const uint32_t batch = task / TILES_PER_BATCH;
            const uint32_t tile = task % TILES_PER_BATCH;
            const uint32_t tileBegin = tile * TOPK_TILE_LEN;
            const uint32_t rowOffset = batch * topk;
            const uint32_t gmOffset = rowOffset + tileBegin;
            AscendC::DataCopy(hitValid, hitValidMaskGm[gmOffset], TOPK_TILE_LEN);
            AscendC::DataCopy(missValid, missValidMaskGm[gmOffset], TOPK_TILE_LEN);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

            uint32_t localHit = 0;
            uint32_t localMiss = 0;
            for (uint32_t i = 0; i < TOPK_TILE_LEN; ++i) {
                if (hitValid.GetValue(i) != 0) {
                    hitLocal.SetValue(localHit++, static_cast<int32_t>(tileBegin + i));
                }
                if (missValid.GetValue(i) != 0) {
                    missLocal.SetValue(localMiss++, static_cast<int32_t>(tileBegin + i));
                }
            }

            const uint32_t expectedHit = static_cast<uint32_t>(tileHitCountsGm.GetValue(task));
            const uint32_t expectedMiss = static_cast<uint32_t>(tileMissCountsGm.GetValue(task));
            const uint32_t hitOffset = static_cast<uint32_t>(tileHitOffsetsGm.GetValue(task));
            const uint32_t missOffset = static_cast<uint32_t>(tileMissOffsetsGm.GetValue(task));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            if (expectedHit > 0) {
                AscendC::DataCopyExtParams copyParams{
                    1, expectedHit * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                AscendC::DataCopyPad(hitSparseIndicesGm[rowOffset + hitOffset],
                                     hitLocal, copyParams);
            }
            if (expectedMiss > 0) {
                AscendC::DataCopyExtParams copyParams{
                    1, expectedMiss * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                AscendC::DataCopyPad(missSparseIndicesGm[rowOffset + missOffset],
                                     missLocal, copyParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
    }

private:
    AscendC::GlobalTensor<uint8_t> hitValidMaskGm;
    AscendC::GlobalTensor<uint8_t> missValidMaskGm;
    AscendC::GlobalTensor<int32_t> tileHitCountsGm;
    AscendC::GlobalTensor<int32_t> tileMissCountsGm;
    AscendC::GlobalTensor<int32_t> tileHitOffsetsGm;
    AscendC::GlobalTensor<int32_t> tileMissOffsetsGm;
    AscendC::GlobalTensor<int32_t> hitSparseIndicesGm;
    AscendC::GlobalTensor<int32_t> missSparseIndicesGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> indexBuf;
    uint32_t batchSize = 0;
    uint32_t topk = 0;
};

extern "C" __global__ __aicore__ void sparse_kv_partition_scatter(
    GM_ADDR hit_valid_mask, GM_ADDR miss_valid_mask, GM_ADDR tile_hit_counts,
    GM_ADDR tile_miss_counts, GM_ADDR tile_hit_offsets, GM_ADDR tile_miss_offsets,
    GM_ADDR hit_sparse_indices, GM_ADDR miss_sparse_indices,
    uint32_t batch_size, uint32_t topk)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSparseKvPartitionScatter kernel;
    kernel.Init(hit_valid_mask, miss_valid_mask, tile_hit_counts, tile_miss_counts,
                tile_hit_offsets, tile_miss_offsets, hit_sparse_indices,
                miss_sparse_indices, batch_size, topk, &pipe);
    kernel.Process();
}
