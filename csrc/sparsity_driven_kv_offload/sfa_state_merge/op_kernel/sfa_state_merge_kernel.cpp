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

#include "kernel_operator.h"

namespace {

constexpr uint32_t kStatsVectorElements = 8;
constexpr float kFloatTiny = 1.1754943508222875e-38F;

enum class PartitionMode : uint32_t {
    kBothEmpty = 0,
    kHitOnly = 1,
    kMissOnly = 2,
    kBothNonEmpty = 3,
};

template <typename T>
class KernelSfaStateMerge
{
public:
    __aicore__ inline KernelSfaStateMerge() {}

    __aicore__ inline void Init(GM_ADDR hitOutput, GM_ADDR hitMax, GM_ADDR hitSum, GM_ADDR missOutput,
                                GM_ADDR missMax, GM_ADDR missSum, GM_ADDR hitCounts, GM_ADDR missCounts,
                                GM_ADDR output, uint32_t batchSize, uint32_t queryLength, uint32_t headCount,
                                uint32_t headDim, uint32_t tileElements, AscendC::TPipe *pipeIn)
    {
        this->batchSize = batchSize;
        this->queryLength = queryLength;
        this->headCount = headCount;
        this->headDim = headDim;
        this->tileElements = tileElements;
        this->pipe = pipeIn;

        const uint64_t rowCount = static_cast<uint64_t>(batchSize) * queryLength * headCount;
        const uint64_t outputElements = rowCount * headDim;
        hitOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hitOutput), outputElements);
        missOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(missOutput), outputElements);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output), outputElements);
        hitMaxGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hitMax), rowCount);
        hitSumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hitSum), rowCount);
        missMaxGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(missMax), rowCount);
        missSumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(missSum), rowCount);
        hitCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(hitCounts), batchSize);
        missCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missCounts), batchSize);

        pipe->InitBuffer(hitQueue, 1, tileElements * sizeof(T));
        pipe->InitBuffer(missQueue, 1, tileElements * sizeof(T));
        pipe->InitBuffer(outputQueue, 1, tileElements * sizeof(T));
        pipe->InitBuffer(hitFloatBuffer, tileElements * sizeof(float));
        pipe->InitBuffer(missFloatBuffer, tileElements * sizeof(float));
        pipe->InitBuffer(expBuffer, kStatsVectorElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t workerCount = AscendC::GetBlockNum();
        const uint32_t workerIndex = AscendC::GetBlockIdx();
        const uint32_t rowCount = batchSize * queryLength * headCount;
        const uint32_t rowsPerWorker = (rowCount + workerCount - 1U) / workerCount;
        const uint32_t rowBegin = workerIndex * rowsPerWorker;
        uint32_t rowEnd = rowBegin + rowsPerWorker;
        if (rowEnd > rowCount) {
            rowEnd = rowCount;
        }

        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            const uint32_t batch = row / (queryLength * headCount);
            const bool hitNonEmpty = hitCountsGm.GetValue(batch) > 0;
            const bool missNonEmpty = missCountsGm.GetValue(batch) > 0;

            PartitionMode mode = PartitionMode::kBothEmpty;
            float hitWeight = 0.0F;
            float missWeight = 0.0F;
            if (hitNonEmpty && missNonEmpty) {
                mode = PartitionMode::kBothNonEmpty;
                ComputeWeights(row, hitWeight, missWeight);
            } else if (hitNonEmpty) {
                mode = PartitionMode::kHitOnly;
                hitWeight = 1.0F;
            } else if (missNonEmpty) {
                mode = PartitionMode::kMissOnly;
                missWeight = 1.0F;
            }

            const uint32_t rowOffset = row * headDim;
            for (uint32_t offset = 0; offset < headDim; offset += tileElements) {
                uint32_t elements = headDim - offset;
                if (elements > tileElements) {
                    elements = tileElements;
                }
                ProcessTile(rowOffset + offset, elements, mode, hitWeight, missWeight);
            }
        }
    }

private:
    __aicore__ inline void ComputeWeights(uint32_t row, float &hitWeight, float &missWeight)
    {
        const float hitMaximum = hitMaxGm.GetValue(row);
        const float missMaximum = missMaxGm.GetValue(row);
        const float globalMaximum = hitMaximum > missMaximum ? hitMaximum : missMaximum;

        AscendC::LocalTensor<float> exponentials = expBuffer.Get<float>();
        for (uint32_t i = 0; i < kStatsVectorElements; ++i) {
            exponentials.SetValue(i, 0.0F);
        }
        exponentials.SetValue(0, hitMaximum - globalMaximum);
        exponentials.SetValue(1, missMaximum - globalMaximum);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Exp(exponentials, exponentials, kStatsVectorElements);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        const float hitMass = hitSumGm.GetValue(row) * exponentials.GetValue(0);
        const float missMass = missSumGm.GetValue(row) * exponentials.GetValue(1);
        const float denominator = hitMass + missMass;
        const float safeDenominator = denominator < kFloatTiny ? kFloatTiny : denominator;
        hitWeight = hitMass / safeDenominator;
        missWeight = missMass / safeDenominator;
    }

    __aicore__ inline void CopyHitInput(uint32_t gmOffset, uint32_t elements)
    {
        AscendC::LocalTensor<T> local = hitQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, elements * sizeof(T), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, hitOutputGm[gmOffset], copyParams, padParams);
        hitQueue.EnQue(local);
    }

    __aicore__ inline void CopyMissInput(uint32_t gmOffset, uint32_t elements)
    {
        AscendC::LocalTensor<T> local = missQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, elements * sizeof(T), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, missOutputGm[gmOffset], copyParams, padParams);
        missQueue.EnQue(local);
    }

    __aicore__ inline void ProcessTile(uint32_t gmOffset, uint32_t elements, PartitionMode mode, float hitWeight,
                                       float missWeight)
    {
        if (mode == PartitionMode::kHitOnly || mode == PartitionMode::kBothNonEmpty) {
            CopyHitInput(gmOffset, elements);
        }
        if (mode == PartitionMode::kMissOnly || mode == PartitionMode::kBothNonEmpty) {
            CopyMissInput(gmOffset, elements);
        }

        AscendC::LocalTensor<T> outputLocal = outputQueue.AllocTensor<T>();
        AscendC::LocalTensor<float> hitFloat = hitFloatBuffer.Get<float>();
        AscendC::LocalTensor<float> missFloat = missFloatBuffer.Get<float>();

        if (mode == PartitionMode::kBothEmpty) {
            AscendC::Duplicate(outputLocal, static_cast<T>(0.0F), elements);
            outputQueue.EnQue(outputLocal);
        } else if (mode == PartitionMode::kHitOnly) {
            AscendC::LocalTensor<T> hitLocal = hitQueue.DeQue<T>();
            AscendC::DataCopy(outputLocal, hitLocal, elements);
            outputQueue.EnQue(outputLocal);
            hitQueue.FreeTensor(hitLocal);
        } else if (mode == PartitionMode::kMissOnly) {
            AscendC::LocalTensor<T> missLocal = missQueue.DeQue<T>();
            AscendC::DataCopy(outputLocal, missLocal, elements);
            outputQueue.EnQue(outputLocal);
            missQueue.FreeTensor(missLocal);
        } else {
            AscendC::LocalTensor<T> hitLocal = hitQueue.DeQue<T>();
            AscendC::LocalTensor<T> missLocal = missQueue.DeQue<T>();
            AscendC::Cast(hitFloat, hitLocal, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::Cast(missFloat, missLocal, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(hitFloat, hitFloat, hitWeight, elements);
            AscendC::Muls(missFloat, missFloat, missWeight, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(hitFloat, hitFloat, missFloat, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, hitFloat, AscendC::RoundMode::CAST_RINT, elements);
            outputQueue.EnQue(outputLocal);
            hitQueue.FreeTensor(hitLocal);
            missQueue.FreeTensor(missLocal);
        }

        AscendC::LocalTensor<T> readyOutput = outputQueue.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, elements * sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[gmOffset], readyOutput, copyParams);
        outputQueue.FreeTensor(readyOutput);
    }

private:
    AscendC::TPipe *pipe = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> hitQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> missQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> hitFloatBuffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> missFloatBuffer;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> expBuffer;

    AscendC::GlobalTensor<T> hitOutputGm;
    AscendC::GlobalTensor<T> missOutputGm;
    AscendC::GlobalTensor<T> outputGm;
    AscendC::GlobalTensor<float> hitMaxGm;
    AscendC::GlobalTensor<float> hitSumGm;
    AscendC::GlobalTensor<float> missMaxGm;
    AscendC::GlobalTensor<float> missSumGm;
    AscendC::GlobalTensor<int32_t> hitCountsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;

    uint32_t batchSize = 0;
    uint32_t queryLength = 0;
    uint32_t headCount = 0;
    uint32_t headDim = 0;
    uint32_t tileElements = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void sfa_state_merge_fp16(
    GM_ADDR hitOutput, GM_ADDR hitMax, GM_ADDR hitSum, GM_ADDR missOutput, GM_ADDR missMax, GM_ADDR missSum,
    GM_ADDR hitCounts, GM_ADDR missCounts, GM_ADDR output, uint32_t batchSize, uint32_t queryLength,
    uint32_t headCount, uint32_t headDim, uint32_t tileElements)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSfaStateMerge<half> kernel;
    kernel.Init(hitOutput, hitMax, hitSum, missOutput, missMax, missSum, hitCounts, missCounts, output, batchSize,
                queryLength, headCount, headDim, tileElements, &pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void sfa_state_merge_bf16(
    GM_ADDR hitOutput, GM_ADDR hitMax, GM_ADDR hitSum, GM_ADDR missOutput, GM_ADDR missMax, GM_ADDR missSum,
    GM_ADDR hitCounts, GM_ADDR missCounts, GM_ADDR output, uint32_t batchSize, uint32_t queryLength,
    uint32_t headCount, uint32_t headDim, uint32_t tileElements)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelSfaStateMerge<bfloat16_t> kernel;
    kernel.Init(hitOutput, hitMax, hitSum, missOutput, missMax, missSum, hitCounts, missCounts, output, batchSize,
                queryLength, headCount, headDim, tileElements, &pipe);
    kernel.Process();
}
