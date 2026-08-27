/**
 * @file unidex_split_copy_kernel.cpp
 * @brief Indexed KV row copy with direct NoPE/RoPE split and optional promotion.
 */

#include "kernel_operator.h"

using CopyUnit = uint8_t;
constexpr uint32_t BUFFER_NUM = 2;

class KernelUniDexSplitCopy
{
public:
    __aicore__ inline KernelUniDexSplitCopy() {}

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dstNope, GM_ADDR dstRope, GM_ADDR hotCache,
                                GM_ADDR srcIndex, GM_ADDR dstIndex, GM_ADDR hotDstIndex, GM_ADDR validMask,
                                uint32_t srcRows, uint32_t dstRows, uint32_t hotRows, uint32_t nopeBytes,
                                uint32_t ropeBytes, uint32_t maxCopy, bool promote, AscendC::TPipe *pipeIn)
    {
        this->srcRows = srcRows;
        this->dstRows = dstRows;
        this->hotRows = hotRows;
        this->nopeBytes = nopeBytes;
        this->ropeBytes = ropeBytes;
        this->rowBytes = nopeBytes + ropeBytes;
        this->maxCopy = maxCopy;
        this->promote = promote;
        this->pipe = pipeIn;

        const uint32_t blockNum = AscendC::GetBlockNum();
        this->copyRowsPerCore = (maxCopy + blockNum - 1U) / blockNum;

        srcGm.SetGlobalBuffer((__gm__ CopyUnit *)src, srcRows * rowBytes);
        dstNopeGm.SetGlobalBuffer((__gm__ CopyUnit *)dstNope, dstRows * nopeBytes);
        dstRopeGm.SetGlobalBuffer((__gm__ CopyUnit *)dstRope, dstRows * ropeBytes);
        if (promote) {
            hotCacheGm.SetGlobalBuffer((__gm__ CopyUnit *)hotCache, hotRows * rowBytes);
            hotDstIndexGm.SetGlobalBuffer((__gm__ int64_t *)hotDstIndex, maxCopy);
        }
        srcIndexGm.SetGlobalBuffer((__gm__ int64_t *)srcIndex, maxCopy);
        dstIndexGm.SetGlobalBuffer((__gm__ int64_t *)dstIndex, maxCopy);
        validMaskGm.SetGlobalBuffer((__gm__ uint8_t *)validMask, maxCopy);

        const uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
        pipe->InitBuffer(copyQue, BUFFER_NUM, alignedRowBytes * sizeof(CopyUnit));
    }

    __aicore__ inline void Process()
    {
        if (rowBytes == 0 || maxCopy == 0) {
            return;
        }

        const uint32_t coreBegin = AscendC::GetBlockIdx() * copyRowsPerCore;
        uint32_t coreEnd = coreBegin + copyRowsPerCore;
        if (coreEnd > maxCopy) {
            coreEnd = maxCopy;
        }

        uint32_t nopeOffsets[BUFFER_NUM] = {0, 0};
        uint32_t ropeOffsets[BUFFER_NUM] = {0, 0};
        uint32_t hotOffsets[BUFFER_NUM] = {0, 0};
        uint32_t queueHead = 0;
        uint32_t queueTail = 0;
        uint32_t queued = 0;

        for (uint32_t i = coreBegin; i < coreEnd; ++i) {
            uint32_t srcOffset = 0;
            uint32_t nopeOffset = 0;
            uint32_t ropeOffset = 0;
            uint32_t hotOffset = 0;
            if (!BuildCopyTask(i, srcOffset, nopeOffset, ropeOffset, hotOffset)) {
                continue;
            }

            if (queued == BUFFER_NUM) {
                CopyOut(nopeOffsets[queueHead], ropeOffsets[queueHead], hotOffsets[queueHead]);
                queueHead = NextQueueIndex(queueHead);
                --queued;
            }

            CopyIn(srcOffset);
            nopeOffsets[queueTail] = nopeOffset;
            ropeOffsets[queueTail] = ropeOffset;
            hotOffsets[queueTail] = hotOffset;
            queueTail = NextQueueIndex(queueTail);
            ++queued;
        }

        while (queued > 0) {
            CopyOut(nopeOffsets[queueHead], ropeOffsets[queueHead], hotOffsets[queueHead]);
            queueHead = NextQueueIndex(queueHead);
            --queued;
        }
    }

private:
    __aicore__ inline uint32_t NextQueueIndex(uint32_t index) const
    {
        return index == BUFFER_NUM - 1U ? 0U : index + 1U;
    }

    __aicore__ inline bool BuildCopyTask(uint32_t mapIdx, uint32_t &srcOffset, uint32_t &nopeOffset,
                                         uint32_t &ropeOffset, uint32_t &hotOffset)
    {
        if (validMaskGm.GetValue(mapIdx) == 0) {
            return false;
        }

        const int64_t srcRow = srcIndexGm.GetValue(mapIdx);
        const int64_t dstRow = dstIndexGm.GetValue(mapIdx);
        if (srcRow < 0 || dstRow < 0 || srcRow >= static_cast<int64_t>(srcRows) ||
            dstRow >= static_cast<int64_t>(dstRows)) {
            return false;
        }

        int64_t hotRow = 0;
        if (promote) {
            hotRow = hotDstIndexGm.GetValue(mapIdx);
            if (hotRow < 0 || hotRow >= static_cast<int64_t>(hotRows)) {
                return false;
            }
        }

        srcOffset = static_cast<uint32_t>(srcRow) * rowBytes;
        nopeOffset = static_cast<uint32_t>(dstRow) * nopeBytes;
        ropeOffset = static_cast<uint32_t>(dstRow) * ropeBytes;
        hotOffset = promote ? static_cast<uint32_t>(hotRow) * rowBytes : 0U;
        return true;
    }

    __aicore__ inline void CopyIn(uint32_t srcOffset)
    {
        AscendC::LocalTensor<CopyUnit> local = copyQue.AllocTensor<CopyUnit>();
        AscendC::DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<CopyUnit> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, srcGm[srcOffset], copyParams, padParams);
        copyQue.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint32_t nopeOffset, uint32_t ropeOffset, uint32_t hotOffset)
    {
        AscendC::LocalTensor<CopyUnit> local = copyQue.DeQue<CopyUnit>();
        AscendC::DataCopyExtParams nopeParams{1, nopeBytes, 0, 0, 0};
        AscendC::DataCopyExtParams ropeParams{1, ropeBytes, 0, 0, 0};
        AscendC::DataCopyPad(dstNopeGm[nopeOffset], local, nopeParams);
        AscendC::DataCopyPad(dstRopeGm[ropeOffset], local[nopeBytes], ropeParams);
        if (promote) {
            AscendC::DataCopyExtParams hotParams{1, rowBytes, 0, 0, 0};
            AscendC::DataCopyPad(hotCacheGm[hotOffset], local, hotParams);
        }
        copyQue.FreeTensor(local);
    }

private:
    AscendC::TPipe *pipe = nullptr;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> copyQue;
    AscendC::GlobalTensor<CopyUnit> srcGm;
    AscendC::GlobalTensor<CopyUnit> dstNopeGm;
    AscendC::GlobalTensor<CopyUnit> dstRopeGm;
    AscendC::GlobalTensor<CopyUnit> hotCacheGm;
    AscendC::GlobalTensor<int64_t> srcIndexGm;
    AscendC::GlobalTensor<int64_t> dstIndexGm;
    AscendC::GlobalTensor<int64_t> hotDstIndexGm;
    AscendC::GlobalTensor<uint8_t> validMaskGm;
    uint32_t srcRows = 0;
    uint32_t dstRows = 0;
    uint32_t hotRows = 0;
    uint32_t nopeBytes = 0;
    uint32_t ropeBytes = 0;
    uint32_t rowBytes = 0;
    uint32_t maxCopy = 0;
    uint32_t copyRowsPerCore = 0;
    bool promote = false;
};

extern "C" __global__ __aicore__ void unidex_split_copy(
    GM_ADDR src, GM_ADDR dst_nope, GM_ADDR dst_rope, GM_ADDR src_index, GM_ADDR dst_index, GM_ADDR valid_mask,
    uint32_t srcRows, uint32_t dstRows, uint32_t nopeBytes, uint32_t ropeBytes, uint32_t maxCopy)
{
    AscendC::TPipe pipe;
    KernelUniDexSplitCopy kernel;
    kernel.Init(src, dst_nope, dst_rope, dst_nope, src_index, dst_index, dst_index, valid_mask, srcRows, dstRows,
                dstRows, nopeBytes, ropeBytes, maxCopy, false, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void unidex_split_copy_promote(
    GM_ADDR src, GM_ADDR dst_nope, GM_ADDR dst_rope, GM_ADDR hot_cache, GM_ADDR src_index, GM_ADDR dst_index,
    GM_ADDR hot_dst_index, GM_ADDR valid_mask, uint32_t srcRows, uint32_t dstRows, uint32_t hotRows,
    uint32_t nopeBytes, uint32_t ropeBytes, uint32_t maxCopy)
{
    AscendC::TPipe pipe;
    KernelUniDexSplitCopy kernel;
    kernel.Init(src, dst_nope, dst_rope, hot_cache, src_index, dst_index, hot_dst_index, valid_mask, srcRows,
                dstRows, hotRows, nopeBytes, ropeBytes, maxCopy, true, &pipe);
    kernel.Process();
}
