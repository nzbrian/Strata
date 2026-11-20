// src/kernels/cuda/iq_mmvq_f32.cu - quantised-weight x FLOAT-activation GEMV for decode.
//
// iq_mmvq (iq_kernels.cu) reads the same quantised weights but takes a Q8_1 activation, the llama.cpp CUDA
// contract.  A Q8_1 activation is not accurate enough over MiniMax-M3's 60 MoE layers: the per-layer error
// compounds to the wrong token (20291 instead of 200058).  This kernel reads the identical quantised weights
// against a FLOAT activation: each block dequantises one weight row to shared memory and dots it in float.
//
// The per-block dequantise functions are ports of src/quant/dequant.cpp's scalar dequant_* (the same ggml block
// layouts and codebook grids).  Only the types the M3 model carries are implemented: Q8_0, Q4_K, Q5_K, Q6_K,
// IQ2_XXS, IQ1_M.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdint>

// The K-quants' 6-bit scale/min decoder (the bit twiddling is the part easy to get wrong), from
// include/strata/artifact/dequant.hpp's get_scale_min_k4, inlined here so this file does not pull in that
// header's host-only includes.
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// The block structs in their C declaration flavour: ggml_half is plain uint16_t and the K-quant scales are
// ordinary members (the header's CUDA flavour wraps them in a union nvcc's C++ mode does not flatten).  The host
// tables it emits here are unused by the device code; the device reads its own __device__ copies.
#define STRATA_QUANT_TABLES
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "../../../third_party/strata-ggml/strata-quant-common.h"

#include "iq_mmvq_f32_tables.inc"   // __device__ codebook tables (values from the same header)

namespace strata::kernels {

// strata/quant/fp16.hpp's fp16_to_fp32 is host-only; the same exact formula with CUDA's bit intrinsics (the
// host version's std::memcpy bit-casts become __uint_as_float / __float_as_uint here).
__device__ __forceinline__ float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const float exp_scale = __uint_as_float(0x7800000u);   // 2^-112
    const float normalized = __uint_as_float((two_w >> 4) + (0xE0u << 23)) * exp_scale;
    const float denormalized = __uint_as_float((two_w >> 17) | (126u << 23)) - 0.5f;
    const uint32_t result = sign | (two_w < (1u << 27) ? __float_as_uint(denormalized) : __float_as_uint(normalized));
    return __uint_as_float(result);
}

// ---- one quantised block -> QK floats.  x is the block, y the QK output slots. ----

__device__ __forceinline__ void dq_q8_0_one(const block_q8_0* x, float* y) {
    const float d = fp16_to_fp32(x->d);
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) y[j] = x->qs[j] * d;
}

__device__ __forceinline__ void dq_q4_K_one(const block_q4_K* x, float* y) {
    const uint8_t* q = x->qs;
    const float d = fp16_to_fp32(x->d);
    const float mn = fp16_to_fp32(x->dmin);
    int is = 0; uint8_t sc, m;
    float* yout = y;   // the CPU original's *y++: advance 64 per iteration or every pass overwrites y[0..63]
#pragma unroll
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, x->scales, sc, m);
        const float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, x->scales, sc, m);
        const float d2 = d * sc, m2 = mn * m;
#pragma unroll
        for (int l = 0; l < 32; ++l) yout[l] = d1 * (q[l] & 0xF) - m1;
#pragma unroll
        for (int l = 0; l < 32; ++l) yout[32 + l] = d2 * (q[l] >> 4) - m2;
        yout += 64;
        q += 32; is += 2;
    }
}

__device__ __forceinline__ void dq_q5_K_one(const block_q5_K* x, float* y) {
    const uint8_t* ql = x->qs;
    const uint8_t* qh = x->qh;
    const float d = fp16_to_fp32(x->d);
    const float mn = fp16_to_fp32(x->dmin);
    int is = 0; uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    float* yout = y;   // as in dq_q4_K_one: the CPU original advances y through the block
#pragma unroll
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, x->scales, sc, m);
        const float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, x->scales, sc, m);
        const float d2 = d * sc, m2 = mn * m;
#pragma unroll
        for (int l = 0; l < 32; ++l) yout[l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
#pragma unroll
        for (int l = 0; l < 32; ++l) yout[32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        yout += 64;
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

__device__ __forceinline__ void dq_q6_K_one(const block_q6_K* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    const uint8_t* ql = x->ql;
    const uint8_t* qh = x->qh;
    const int8_t* sc = x->scales;
#pragma unroll
    for (int n = 0; n < QK_K; n += 128) {
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

__device__ __forceinline__ void dq_iq2_xxs_one(const block_iq2_xxs* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*) aux32;
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        // byte assembly, not a uint32_t load: the block need not be 4-byte aligned in the row.  qs is
        // uint16_t[QK_K/8], so the ib32 group of 4 quants (2 uint32) starts at uint16 index 4*ib32 = byte 8*ib32
        const uint8_t* qb = (const uint8_t*) (const void*) x->qs + 8 * ib32;
        aux32[0] = (uint32_t) qb[0] | ((uint32_t) qb[1] << 8) | ((uint32_t) qb[2] << 16) | ((uint32_t) qb[3] << 24);
        aux32[1] = (uint32_t) qb[4] | ((uint32_t) qb[5] << 8) | ((uint32_t) qb[6] << 16) | ((uint32_t) qb[7] << 24);
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int8_t* grid = (const int8_t*) (dev_iq2xxs_grid + aux8[l]);
            const uint8_t signs = dev_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                y[ib32 * 32 + l * 8 + j] = db * grid[j] * (signs & dev_kmask_iq2xs[j] ? -1.f : 1.f);
        }
    }
}

__device__ __forceinline__ void dq_iq1_m_one(const block_iq1_m* x, float* y) {
    // byte assembly for the four fp16 scales (the block need not be 2-byte aligned in the row)
    const uint8_t* sb = (const uint8_t*) (const void*) x->scales;   // ggml_half[] in the C flavour
    const uint16_t sc[4] = {
        (uint16_t) sb[0] | ((uint16_t) sb[1] << 8), (uint16_t) sb[2] | ((uint16_t) sb[3] << 8),
        (uint16_t) sb[4] | ((uint16_t) sb[5] << 8), (uint16_t) sb[6] | ((uint16_t) sb[7] << 8),
    };
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = fp16_to_fp32(scale.f16);
    const uint8_t* qs = x->qs;
    const uint8_t* qh = x->qh;
#pragma unroll
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        const uint16_t idx0 = qs[0] | ((qh[0] << 8) & 0x700);
        const uint16_t idx1 = qs[1] | ((qh[0] << 4) & 0x700);
        const uint16_t idx2 = qs[2] | ((qh[1] << 8) & 0x700);
        const uint16_t idx3 = qs[3] | ((qh[1] << 4) & 0x700);
        const float del0 = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        const float del1 = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        const float del2 = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        const float del3 = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        const int8_t* g0 = (const int8_t*) (dev_iq1s_grid + idx0);
        const int8_t* g1 = (const int8_t*) (dev_iq1s_grid + idx1);
        const int8_t* g2 = (const int8_t*) (dev_iq1s_grid + idx2);
        const int8_t* g3 = (const int8_t*) (dev_iq1s_grid + idx3);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int o = ib * 32 + j;
            y[o +  0] = dl1 * (g0[j] + del0);
            y[o +  8] = dl1 * (g1[j] + del1);
            y[o + 16] = dl2 * (g2[j] + del2);
            y[o + 24] = dl2 * (g3[j] + del3);
        }
        qs += 4; qh += 2;
    }
}

__device__ __forceinline__ void dq_iq2_xs_one(const block_iq2_xs* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    const uint16_t* qs = x->qs;   // uint16 units: low 9 bits = grid index, bits 9..15 = sign index
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const float db0 = d * (0.5f + (x->scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (x->scales[ib32] >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint16_t q = qs[4 * ib32 + l];
            const int8_t* grid = (const int8_t*) (dev_iq2xs_grid + (q & 511));
            const uint8_t signs = dev_ksigns_iq2xs[q >> 9];
            const float sdb = (l < 2) ? db0 : db1;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                y[ib32 * 32 + l * 8 + j] = sdb * grid[j] * (signs & dev_kmask_iq2xs[j] ? -1.f : 1.f);
        }
    }
}

// dequantise block `bi` of the row at `wrow` into `sw + bi*QK`.  Returns the type's QK (0 for an unsupported id).
__device__ int dequant_block(int type, const uint8_t* wrow, int bi, float* sw) {
    switch (type) {
        case 8:  { const auto* b = (const block_q8_0*)    wrow; dq_q8_0_one   (&b[bi], sw + bi * QK8_0); return QK8_0; }
        case 12: { const auto* b = (const block_q4_K*)    wrow; dq_q4_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 13: { const auto* b = (const block_q5_K*)    wrow; dq_q5_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 14: { const auto* b = (const block_q6_K*)    wrow; dq_q6_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 16: { const auto* b = (const block_iq2_xxs*) wrow; dq_iq2_xxs_one(&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 17: { const auto* b = (const block_iq2_xs*)  wrow; dq_iq2_xs_one (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 29: { const auto* b = (const block_iq1_m*)   wrow; dq_iq1_m_one  (&b[bi], sw + bi * QK_K);  return QK_K;  }
        default: return 0;
    }
}

// y[n] = W[n] . x for n in [0, n_out): W is n_out rows of n_in quantised values (row `row_bytes`), x a float
// activation of n_in.  One block per output row; the row is dequantised to shared memory, then dotted in float.
// (x is the same for every row and stays L2-resident.)
extern "C" __global__ void __launch_bounds__(256)
mmvq_f32_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int type,
                const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int n = blockIdx.x;
    if (n >= n_out) return;
    extern __shared__ float sw[];
    const uint8_t* wrow = w + (size_t) n * row_bytes;
    const int qk = (type == 8) ? QK8_0 : QK_K;
    const int nb = n_in / qk;
    for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) dequant_block(type, wrow, bi, sw);
    __syncthreads();
    float acc = 0.f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += sw[i] * x[i];
    __shared__ float warp_sum[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x >> 5)) ? warp_sum[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[n] = v;
    }
}

void iq_mmvq_f32(int ggml_type, const void* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    const size_t row_bytes = iq_row_bytes(ggml_type, n_in);
    if (row_bytes == 0) return;   // unsupported type or n_in not a whole number of blocks
    const size_t shmem = (size_t) n_in * sizeof(float);
    mmvq_f32_kernel<<<n_out, 256, shmem, (cudaStream_t) stream>>>(
        (const uint8_t*) w, row_bytes, ggml_type, x, y, n_in, n_out);
}

}  // namespace strata::kernels
