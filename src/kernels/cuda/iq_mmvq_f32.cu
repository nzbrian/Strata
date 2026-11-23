#include <cuda_fp16.h>
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
#include <cstdio>
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

__device__ __forceinline__ void dq_iq3_xxs_one(const block_iq3_xxs* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    const uint8_t* qs = x->qs;
    const uint8_t* sas = qs + QK_K / 4;   // the last 32 bytes carry scales and sign indices
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            // byte assembly: the 98-byte block stride leaves sas unaligned for a uint32 load
            const uint8_t* sb = sas + 4 * ib32;
            const uint32_t aux32 = (uint32_t) sb[0] | ((uint32_t) sb[1] << 8) | ((uint32_t) sb[2] << 16) | ((uint32_t) sb[3] << 24);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = dev_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const int8_t* g1 = (const int8_t*) (dev_iq3xxs_grid + qs[2 * l + 0]);
            const int8_t* g2 = (const int8_t*) (dev_iq3xxs_grid + qs[2 * l + 1]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                y[ib32 * 32 + l * 8 + j + 0] = db * g1[j] * (signs & dev_kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[ib32 * 32 + l * 8 + j + 4] = db * g2[j] * (signs & dev_kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
        }
        qs += 8;
    }
}

__device__ __forceinline__ void dq_iq4_xs_one(const block_iq4_xs* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    const uint8_t* qs = x->qs;
#pragma unroll
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = ((x->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            y[ib * 32 + j + 0] = dl * dev_kvalues_iq4nl[qs[j] & 0xf];
            y[ib * 32 + j + 16] = dl * dev_kvalues_iq4nl[qs[j] >> 4];
        }
        qs += 16;
    }
}

// ggml's ue4m3 (NVFP4 sub-block scales) decoder, halved as ggml does (src/quant/internal.hpp's ue4m3_formula).
__device__ __forceinline__ float ue4m3_dev(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF, man = x & 0x7;
    const float raw = exp == 0 ? exp2f(-9.0f) * (float) man : exp2f((float) exp - 7.0f) * (1.0f + (float) man / 8.0f);
    return raw * 0.5f;
}

__device__ __forceinline__ void dq_nvfp4_one(const block_nvfp4* x, float* y) {
#pragma unroll
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
        const float d = ue4m3_dev(x->d[s]);
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const int v0 = dev_kvalues_fp4[x->qs[s * (QK_NVFP4_SUB / 2) + j] & 0x0F];
            const int v1 = dev_kvalues_fp4[x->qs[s * (QK_NVFP4_SUB / 2) + j] >> 4];
            y[s * QK_NVFP4_SUB + j + 0] = v0 * d;
            y[s * QK_NVFP4_SUB + j + QK_NVFP4_SUB / 2] = v1 * d;
        }
    }
}

__device__ __forceinline__ void dq_q2_K_one(const block_q2_K* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    const float mn = fp16_to_fp32(x->dmin);
    const uint8_t* q = x->qs;
    int is = 0;
#pragma unroll
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t sc = x->scales[is++];
            const float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
#pragma unroll
            for (int l = 0; l < 16; ++l) y[n + l] = dl * ((int8_t) ((q[l] >> shift) & 3)) - ml;
            const uint8_t sc2 = x->scales[is++];
            const float dl2 = d * (sc2 & 0xF), ml2 = mn * (sc2 >> 4);
#pragma unroll
            for (int l = 0; l < 16; ++l) y[n + 16 + l] = dl2 * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml2;
            shift += 2;
        }
        q += 32;
    }
}

__device__ __forceinline__ void dq_q3_K_one(const block_q3_K* x, float* y) {
    const float d = fp16_to_fp32(x->d);
    const uint8_t* q = x->qs;
    const uint8_t* hm = x->hmask;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*) aux;
    {
        const uint8_t* sb = x->scales;
        aux[0] = (uint32_t) sb[0] | ((uint32_t) sb[1] << 8) | ((uint32_t) sb[2] << 16) | ((uint32_t) sb[3] << 24);
        aux[1] = (uint32_t) sb[4] | ((uint32_t) sb[5] << 8) | ((uint32_t) sb[6] << 16) | ((uint32_t) sb[7] << 24);
        aux[2] = (uint32_t) sb[8] | ((uint32_t) sb[9] << 8) | ((uint32_t) sb[10] << 16) | ((uint32_t) sb[11] << 24);
    }
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int is = 0;
    uint8_t m = 1;
#pragma unroll
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float dl = d * (scales[is++] - 32);
#pragma unroll
            for (int l = 0; l < 16; ++l) y[n + l] = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            const float dl2 = d * (scales[is++] - 32);
#pragma unroll
            for (int l = 0; l < 16; ++l) y[n + 16 + l] = dl2 * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
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
        case 10: { const auto* b = (const block_q2_K*)    wrow; dq_q2_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 11: { const auto* b = (const block_q3_K*)    wrow; dq_q3_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 12: { const auto* b = (const block_q4_K*)    wrow; dq_q4_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 13: { const auto* b = (const block_q5_K*)    wrow; dq_q5_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 14: { const auto* b = (const block_q6_K*)    wrow; dq_q6_K_one   (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 16: { const auto* b = (const block_iq2_xxs*) wrow; dq_iq2_xxs_one(&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 17: { const auto* b = (const block_iq2_xs*)  wrow; dq_iq2_xs_one (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 18: { const auto* b = (const block_iq3_xxs*) wrow; dq_iq3_xxs_one(&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 23: { const auto* b = (const block_iq4_xs*)  wrow; dq_iq4_xs_one (&b[bi], sw + bi * QK_K);  return QK_K;  }
        case 40: { const auto* b = (const block_nvfp4*)   wrow; dq_nvfp4_one  (&b[bi], sw + bi * QK_NVFP4); return QK_NVFP4; }
        case 29: { const auto* b = (const block_iq1_m*)   wrow; dq_iq1_m_one  (&b[bi], sw + bi * QK_K);  return QK_K;  }
        default: return 0;
    }
}

// y[n] = W[n] . x for n in [0, n_out): W is n_out rows of n_in quantised values (row `row_bytes`), x a float
// activation of n_in.  One block per output row; the row is dequantised to shared memory, then dotted in float.
// (x is the same for every row and stays L2-resident.)
extern "C" __global__ void __launch_bounds__(512)
mmvq_f32_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int type,
                const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    // Row-parallel: SPLIT blocks share one output row (block `r*SPLIT+p` owns input range [p*seg, (p+1)*seg)).
    // A whole row in F32 shared memory (48 KB at the dense width) pinned one block per SM at 25% thread
    // occupancy; a third of a row (16 KB) lets six blocks co-reside.  Partials merge with atomicAdd.
    const int SPLIT = 1;
    const int n = blockIdx.x / SPLIT;
    const int p = blockIdx.x % SPLIT;
    if (n >= n_out) return;
    extern __shared__ float sw[];
    const int qk = (type == 8) ? QK8_0 : (type == 40) ? QK_NVFP4 : QK_K;
    const int seg = n_in / SPLIT;              // multiple of every QK for the M3 widths (6144, 12288, ...)
    const int i0 = p * seg;
    const uint8_t* wrow = w + (size_t) n * row_bytes;
    const int nb = seg / qk;
    // dequant_block writes block `b` at `out + b*qk`; hand it a base shifted so segment block 0 lands at sw[0]
    for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) dequant_block(type, wrow, i0 / qk + bi, sw - (i0 / qk) * qk);
    __syncthreads();
    float acc = 0.f;
    for (int i = threadIdx.x; i < seg; i += blockDim.x) acc += sw[i] * x[i0 + i];
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
    if (ggml_type == 17 || ggml_type == 18) { iq_mmvq_f32_fused(ggml_type, w, x, y, n_in, n_out, stream); return; }
    const size_t row_bytes = iq_row_bytes(ggml_type, n_in);
    if (row_bytes == 0) return;   // unsupported type or n_in not a whole number of blocks
    const int SPLIT = 1;
    const size_t shmem = (size_t) (n_in / SPLIT) * sizeof(float);
    static bool opted = false;
    if (!opted && shmem >= 48u << 10) { cudaFuncSetAttribute(mmvq_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); opted = true; }
    mmvq_f32_kernel<<<n_out * SPLIT, 512, shmem, (cudaStream_t) stream>>>(
        (const uint8_t*) w, row_bytes, ggml_type, x, y, n_in, n_out);
}


// Fused dequant-and-dot for the expert-heavy types (IQ2_XS gate/up, IQ3_XXS down in the UD-Q*_XL mixes):
// no shared memory, no barriers - 128 threads per row stride over 32-value sub-blocks, dequantising inline
// and accumulating against the F32 activation.  The smem kernel above stays for the other types.
__device__ __forceinline__ float fused_dot_iq2xs_sub(const block_iq2_xs* blk, int ib32, const float* x) {
    const float d = fp16_to_fp32(blk->d);
    const float db0 = d * (0.5f + (blk->scales[ib32] & 0xf)) * 0.25f;
    const float db1 = d * (0.5f + (blk->scales[ib32] >> 4)) * 0.25f;
    const uint16_t* qs = blk->qs + 4 * ib32;
    float acc = 0.f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint16_t q = qs[l];
        const int8_t* grid = (const int8_t*) (dev_iq2xs_grid + (q & 511));
        const uint8_t signs = dev_ksigns_iq2xs[q >> 9];
        const float sdb = (l < 2) ? db0 : db1;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += sdb * (signs & dev_kmask_iq2xs[j] ? -grid[j] : grid[j]) * x[l * 8 + j];
    }
    return acc;
}

__device__ __forceinline__ float fused_dot_iq3xxs_sub(const block_iq3_xxs* blk, int ib32, const float* x) {
    const float d = fp16_to_fp32(blk->d);
    const uint8_t* qs = blk->qs + 8 * ib32;
    const uint8_t* sb = blk->qs + QK_K / 4 + 4 * ib32;   // sas, byte-wise (unaligned on the 98-byte stride)
    const uint32_t aux32 = (uint32_t) sb[0] | ((uint32_t) sb[1] << 8) | ((uint32_t) sb[2] << 16) | ((uint32_t) sb[3] << 24);
    const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
    float acc = 0.f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t signs = dev_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const int8_t* g1 = (const int8_t*) (dev_iq3xxs_grid + qs[2 * l + 0]);
        const int8_t* g2 = (const int8_t*) (dev_iq3xxs_grid + qs[2 * l + 1]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += db * (signs & dev_kmask_iq2xs[j + 0] ? -g1[j] : g1[j]) * x[l * 8 + j + 0];
            acc += db * (signs & dev_kmask_iq2xs[j + 4] ? -g2[j] : g2[j]) * x[l * 8 + j + 4];
        }
    }
    return acc;
}

template <typename BlockT, float (*DOT)(const BlockT*, int, const float*)>
__global__ void __launch_bounds__(128) mmvq_f32_fused_kernel(const BlockT* __restrict__ w,
                                                             const float* __restrict__ x, float* __restrict__ y,
                                                             int n_sub, int n_blk, int n_out) {
    const int n = blockIdx.x;
    if (n >= n_out) return;
    const BlockT* wrow = w + (size_t) n * n_blk;           // n_blk 256-value blocks per row
    float acc = 0.f;
    for (int s = threadIdx.x; s < n_sub; s += blockDim.x)
        acc += DOT(wrow + s / 8, s % 8, x + s * 32);
    __shared__ float warp_sum[4];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = warp_sum[0] + warp_sum[1] + warp_sum[2] + warp_sum[3];
        y[n] = v;
    }
}

void iq_mmvq_f32_fused(int type, const void* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    if (type == 17) {
        mmvq_f32_fused_kernel<block_iq2_xs, fused_dot_iq2xs_sub><<<n_out, 128, 0, (cudaStream_t) stream>>>(
            (const block_iq2_xs*) w, x, y, n_in / 32, n_in / 256, n_out);
    } else if (type == 18) {
        mmvq_f32_fused_kernel<block_iq3_xxs, fused_dot_iq3xxs_sub><<<n_out, 128, 0, (cudaStream_t) stream>>>(
            (const block_iq3_xxs*) w, x, y, n_in / 32, n_in / 256, n_out);
    }
}


// ===========================================================================
// Chunk GEMMs: Y[M][n_out] = X[M][n_in] . W[n_out][n_in]^T  (prefill).
//
// One block per output row: the row is dequantised to shared memory ONCE, then
// dotted against every one of the M activation rows.  The weights are read
// once per layer-chunk instead of once per token - that is the whole point of
// chunked prefill (decode reads them C times as many).
extern "C" __global__ void __launch_bounds__(512)
mmq_f32_chunk_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int type,
                     const float* __restrict__ x, float* __restrict__ y,
                     int n_in, int n_out, int M) {
    const int n = blockIdx.x;
    if (n >= n_out) return;
    extern __shared__ float sw[];
    const uint8_t* wrow = w + (size_t) n * row_bytes;
    const int qk = (type == 8) ? QK8_0 : (type == 40) ? QK_NVFP4 : QK_K;
    const int nb = n_in / qk;
    for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) dequant_block(type, wrow, bi, sw);
    __syncthreads();
    __shared__ float warp_sum[16];
    for (int m = 0; m < M; ++m) {
        const float* xr = x + (size_t) m * n_in;
        float acc = 0.f;
        for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += sw[i] * xr[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x >> 5)) ? warp_sum[threadIdx.x] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) y[(size_t) m * n_out + n] = v;
        }
        __syncthreads();
    }
}

// F32-weighted variant (the small projections: idxq/idxk/router/head): the row is kept in shared memory
// too - it is re-read M times, and shared beats L1 for that.
extern "C" __global__ void __launch_bounds__(512)
mmq_f32_chunk_f32w_kernel(const float* __restrict__ w,
                          const float* __restrict__ x, float* __restrict__ y,
                          int n_in, int n_out, int M) {
    const int n = blockIdx.x;
    if (n >= n_out) return;
    extern __shared__ float sw[];
    const float* wrow = w + (size_t) n * n_in;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) sw[i] = wrow[i];
    __syncthreads();
    __shared__ float warp_sum[16];
    for (int m = 0; m < M; ++m) {
        const float* xr = x + (size_t) m * n_in;
        float acc = 0.f;
        for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += sw[i] * xr[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x >> 5)) ? warp_sum[threadIdx.x] : 0.f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) y[(size_t) m * n_out + n] = v;
        }
        __syncthreads();
    }
}

void iq_mmq_f32_chunk(int ggml_type, const void* w, const float* x, float* y,
                      int n_in, int n_out, int M, void* stream) {
    const size_t row_bytes = iq_row_bytes(ggml_type, n_in);
    if (row_bytes == 0) return;
    size_t shmem = (size_t) n_in * sizeof(float);
    static bool opted = false;
    if (!opted) {
        // Opt in BELOW the advertised max: on sm_120 sharedMemPerBlockOptin is 101376, but a
        // 512-thread/110-register kernel can only get ~100000 (the driver folds register usage
        // in). Requesting 101376 fails with invalid argument, the limit stays at the 48 KB
        // default, and a 48 KB-dynamic + 64 B-static launch then dies with invalid argument
        // (zero output) - seen with n_in=12288.
        cudaFuncSetAttribute(mmq_f32_chunk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 100000);
        opted = true;
    }    mmq_f32_chunk_kernel<<<n_out, 512, shmem, (cudaStream_t) stream>>>(
        (const uint8_t*) w, row_bytes, ggml_type, x, y, n_in, n_out, M);
}

void iq_mmq_f32_chunk_f32w(const float* w, const float* x, float* y,
                           int n_in, int n_out, int M, void* stream) {
    const size_t shmem = (size_t) n_in * sizeof(float);
    static bool opted = false;
    if (!opted && shmem >= 48u << 10) {
        cudaFuncSetAttribute(mmq_f32_chunk_f32w_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
        opted = true;
    }
    mmq_f32_chunk_f32w_kernel<<<n_out, 512, shmem, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, M);
}

// Batched GEMV: E experts (same type/shape, weight bases in `w[E]`) in one launch.  grid = E*n_out,
// block (e, r) dots row r of expert e.  Lets a layer's four routed experts be copied in first and then
// computed back-to-back with no copy/compute serialization between experts.
extern "C" __global__ void __launch_bounds__(512)
mmvq_f32_batch(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
               const uint8_t* __restrict__ w2, const uint8_t* __restrict__ w3,
               size_t row_bytes, int type,
               const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out, int E, int per_expert_x) {
    const int e = blockIdx.x / n_out;
    const int n = blockIdx.x % n_out;
    const uint8_t* wrow = (e == 0 ? w0 : e == 1 ? w1 : e == 2 ? w2 : w3) + (size_t) n * row_bytes;
    extern __shared__ float sw[];
    const int qk = (type == 8) ? QK8_0 : (type == 40) ? QK_NVFP4 : QK_K;
    const int nb = n_in / qk;
    for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) dequant_block(type, wrow, bi, sw);
    __syncthreads();
    // per_expert_x: expert e reads its OWN input row x + e*n_in (the down projection, where each
    // expert's swiglu output differs); otherwise all experts share x (gate/up on the same dh).
    const float* xe = per_expert_x ? x + (size_t) e * n_in : x;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += sw[i] * xe[i];
    __shared__ float warp_sum[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x >> 5)) ? warp_sum[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[(size_t) e * n_out + n] = v;
    }
}

void iq_mmvq_f32_batch(int E, const void* const w[], int ggml_type, int n_in, int n_out,
                       const float* x, float* y, void* stream, int per_expert_x) {
    const size_t row_bytes = iq_row_bytes(ggml_type, n_in);
    if (row_bytes == 0) return;
    const size_t shmem = (size_t) n_in * sizeof(float);
    static bool opted = false;
    if (!opted && shmem >= 48u << 10) { cudaFuncSetAttribute(mmvq_f32_batch, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem); opted = true; }
    const uint8_t* w0 = E > 0 ? (const uint8_t*) w[0] : nullptr;
    const uint8_t* w1 = E > 1 ? (const uint8_t*) w[1] : nullptr;
    const uint8_t* w2 = E > 2 ? (const uint8_t*) w[2] : nullptr;
    const uint8_t* w3 = E > 3 ? (const uint8_t*) w[3] : nullptr;
    mmvq_f32_batch<<<(size_t) E * n_out, 512, shmem, (cudaStream_t) stream>>>(
        w0, w1, w2, w3, row_bytes, ggml_type, x, y, n_in, n_out, E, per_expert_x);
}

}  // namespace strata::kernels
