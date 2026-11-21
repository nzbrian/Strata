// src/minimax/m3_cuda.cuh - the CUDA kernels for the MiniMax-M3.1-Flash engine.
//
// Straightforward, correct kernels: the same math as the CPU forward (include/strata/minimax/forward.hpp, which
// matches the reference). Projections go through the existing quantised GEMM (iq_mmvq on Q8_1 activations); the
// elementwise and attention work is here. No batching, no graphs - a decode-shaped forward that must produce the
// same token as the CPU forward.
#pragma once
#include <cuda_fp16.h>

namespace m3k {

/// x = rms(y) * w, one row of n.
__global__ void k_rms(const float* __restrict__ y, const float* __restrict__ w, float* __restrict__ x, int n, float eps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float* yv = y + (size_t) blockIdx.y * n;
    float* xv = x + (size_t) blockIdx.y * n;
    if (i >= n) return;
    float ss = 0.0f;
    for (int j = 0; j < n; ++j) { float v = yv[j]; ss += v * v; }
    const float r = rsqrtf(ss / n + eps);
    xv[i] = yv[i] * r * (w ? w[i] : 1.0f);
}

/// Add: x += y, one row of n.
__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[(size_t) blockIdx.y * n + i] += y[i];
}

/// Partial RoPE, NEOX pairing, in place, on q (H heads) and k (HKV heads), head_dim HD, nrot pairs. t is the
/// position, base the frequency base. Matches strata::minimax::rope_partial.
__global__ void k_rope(float* __restrict__ q, float* __restrict__ k, int H, int HKV, int HD, int nrot, float t, float base) {
    const int head = blockIdx.x;      // 0..H+HKV-1: 0..H-1 are q heads, H..H+HKV-1 are k heads
    float* v = (head < H ? q + (size_t) head * HD : k + (size_t) (head - H) * HD);
    if (threadIdx.x >= nrot / 2) return;
    const int i = threadIdx.x;        // one thread per pair (i, i + nrot/2)
    const float fi = (float) i;
    const float ang = t * exp2f(-2.0f * fi / nrot * log2f(base));   // t * base^(-2i/nrot)
    const float cos = __cosf(ang), sin = __sinf(ang);
    const float a = v[i], b = v[i + nrot / 2];
    v[i] = a * cos - b * sin;
    v[i + nrot / 2] = b * cos + a * sin;
}

/// Append one position's K (or V) to the [heads, cap, HD] cache: for each KV head h, cache[h][pos] = src[h].
__global__ void k_kv_append(float* __restrict__ cache, const float* __restrict__ src, int HKV, int HD, int cap, int pos) {
    const int h = blockIdx.x;
    if (h >= HKV) return;
    float* dst = cache + (size_t) h * cap * HD + (size_t) pos * HD;
    const float* s = src + (size_t) h * HD;
    for (int j = threadIdx.x; j < HD; j += blockDim.x) dst[j] = s[j];
}

/// GQA causal attention, one query row (H heads) against the visible positions 0..qpos, out = H x HD. cap is the KV
/// row stride (the cache capacity). One thread per output element, an online softmax in double - the same order of
/// operations as strata::minimax::attn_one, so it matches the CPU recipe.
__global__ void k_attn(const float* __restrict__ q, const float* __restrict__ kv, const float* __restrict__ vv,
                       float* __restrict__ out, int H, int HKV, int HD, int npos, int cap, int qpos) {
    const int h = blockIdx.x;
    if (h >= H || threadIdx.x >= HD) return;
    const int hkv = h / (H / HKV);
    const float* qh = q + (size_t) h * HD;
    const float* kh = kv + (size_t) hkv * cap * HD;
    const float* vh = vv + (size_t) hkv * cap * HD;
    const int nvis = qpos + 1;
    const float scale = rsqrtf((float) HD);
    double m = -1e300, l = 0.0, acc = 0.0;
    for (int p = 0; p < nvis; ++p) {
        const float* kp = kh + (size_t) p * HD;
        double s = 0.0;
        for (int j = 0; j < HD; ++j) s += (double) qh[j] * kp[j];
        s *= scale;
        const double nm = s > m ? s : m;
        const double corr = (m == -1e300) ? 0.0 : exp(m - nm);
        const double w = exp(s - nm);
        m = nm;
        acc = acc * corr + w * vh[(size_t) p * HD + threadIdx.x];
        l = l * corr + w;
    }
    out[(size_t) h * HD + threadIdx.x] = (float) (acc * (l > 0.0 ? 1.0 / l : 0.0));
}

/// y[o] = sum_i W[o* n_in + i] * x[i], F32 weight (for the few F32 projections, e.g. the k/v of M3).
__global__ void k_f32_mm(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    const float* row = W + (size_t) o * n_in;
    double acc = 0.0;   // double accumulation, matching the CPU reference's `linear` (F32 accum drifts over 60 layers)
    for (int i = 0; i < n_in; ++i) acc += (double) row[i] * (double) x[i];
    y[o] = (float) acc;
}

/// x += s * y, one row of n.
__global__ void k_scale_add(float* __restrict__ x, const float* __restrict__ y, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = (size_t) blockIdx.y * n;
    if (i >= n) return;
    x[base + i] += s * y[i];
}

/// x *= s, one row of n.
__global__ void k_scale(float* __restrict__ x, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = (size_t) blockIdx.y * n;
    if (i >= n) return;
    x[base + i] *= s;
}

/// MSA attention: GQA over an irregular set of visible positions (the indexer's selected blocks), one query row
/// (H heads). `vis` is HKV*npos bytes: `vis[hkv*npos+p]` is 1 when GQA group `hkv` may read position `p`.  Each
/// group reads only its OWN indexer head's selected blocks (llama.cpp gathers `tr = cell*HKV + h`), NOT the union,
/// so one shared mask would attend 4x too many keys and corrupt longer prompts.  One thread per output element,
/// an online softmax in double.
__global__ void k_msa_attn(const float* __restrict__ q, const float* __restrict__ kv, const float* __restrict__ vv,
                           const unsigned char* __restrict__ vis, float* __restrict__ out,
                           int H, int HKV, int HD, int npos, int cap) {
    const int h = blockIdx.x;
    if (h >= H || threadIdx.x >= HD) return;
    const int hkv = h / (H / HKV);
    const float* qh = q + (size_t) h * HD;
    const float* kh = kv + (size_t) hkv * cap * HD;
    const float* vh = vv + (size_t) hkv * cap * HD;
    const float scale = rsqrtf((float) HD);
    double m = -1e300, l = 0.0, acc = 0.0;
    for (int p = 0; p < npos; ++p) {
        if (!vis[(size_t) hkv * npos + p]) continue;
        const float* kp = kh + (size_t) p * HD;
        double s = 0.0;
        for (int j = 0; j < HD; ++j) s += (double) qh[j] * kp[j];
        s *= scale;
        const double nm = s > m ? s : m;
        const double corr = (m == -1e300) ? 0.0 : exp(m - nm);
        const double w = exp(s - nm);
        m = nm;
        acc = acc * corr + w * vh[(size_t) p * HD + threadIdx.x];
        l = l * corr + w;
    }
    out[(size_t) h * HD + threadIdx.x] = (float) (acc * (l > 0.0 ? 1.0 / l : 0.0));
}

/// swiglu_oai: h = min(gate,7)*sigmoid(1.702*min(gate,7)) * (1 + clamp(up,-7,7)), one row of nff.
__global__ void k_swiglu_oai(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h, int nff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = (size_t) blockIdx.y * nff;
    if (i >= nff) return;
    const float g = fminf(gate[base + i], 7.0f);
    const float u = fminf(fmaxf(up[base + i], -7.0f), 7.0f);
    const float sig = 1.0f / (1.0f + __expf(-1.702f * g));   // sigmoid(1.702 * g)
    h[base + i] = g * sig * (1.0f + u);
}

/// Convert one KV block of F16 (pinned-RAM staging, H2D'd) to the F32 pool slot. n = blk*HKV*HD.
__global__ void k_f16_to_f32(const __half* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

/// MSA attention over a per-layer BLOCK POOL (F16, the RAM bytes H2D'd verbatim): the host selected top-k
/// blocks per GQA group (the union of groups always fits the pool) and built the per-group pool mask.  `sel`
/// is n_groups*B bytes: sel[g*B+b] = 1 when group g attends to pool block b.  `last_slot`/`last_valid` mark
/// the pool slot holding the in-progress block (positions past last_valid are padding) and -1 when none.
/// One thread per output element, online softmax in double, same recipe as k_msa_attn.
__global__ void k_msa_attn_pool(const float* __restrict__ q, const float* __restrict__ pk, const float* __restrict__ pv,
                                const unsigned char* __restrict__ sel, float* __restrict__ out,
                                int H, int HKV, int HD, int B, int blk, int last_slot, int last_valid) {
    // Flash-decode style: 128 threads per head split the B*blk (block,position) pairs; scores and weights
    // live in dynamic shared memory (64KB - opt-in required). F32 throughout: F64 on consumer cards runs at
    // 1/64 rate and this kernel is on the hot path of every decode step.
    extern __shared__ float smem[];
    const int N = B * blk;
    float* s_s = smem;          // [N] scores
    float* s_w = smem + N;      // [N] softmax weights
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int hkv = h / (H / HKV);
    const float* qh = q + (size_t) h * HD;
    const float scale = rsqrtf((float) HD);
    for (int i = t; i < N; i += 128) {
        const int b = i / blk, p = i - b * blk;
        if (!sel[(size_t) hkv * B + b] || p >= (b == last_slot ? last_valid : blk)) { s_s[i] = -1e30f; continue; }
        const float* krow = pk + ((size_t) b * blk + p) * HKV * HD + (size_t) hkv * HD;
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < HD; ++j) s += qh[j] * krow[j];
        s_s[i] = s * scale;
    }
    __syncthreads();
    // block max
    float mx = -1e30f;
    for (int i = t; i < N; i += 128) mx = fmaxf(mx, s_s[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((t & 31) == 0) smem[N + 2 * (t >> 5)] = mx;
    __syncthreads();
    if (t == 0) {
        float m2 = smem[N + 2 * 0];
        #pragma unroll
        for (int w2 = 1; w2 < 4; ++w2) m2 = fmaxf(m2, smem[N + 2 * w2]);
        smem[N + 2 * 0] = m2;
    }
    __syncthreads();
    mx = smem[N + 2 * 0];
    // weights + block sum
    float ls = 0.f;
    for (int i = t; i < N; i += 128) { const float w = (s_s[i] < -1e29f) ? 0.f : expf(s_s[i] - mx); s_w[i] = w; ls += w; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
    if ((t & 31) == 0) smem[N + 2 * 1 + (t >> 5)] = ls;
    __syncthreads();
    if (t == 0) {
        float l2 = smem[N + 2 * 1];
        #pragma unroll
        for (int w2 = 1; w2 < 4; ++w2) l2 += smem[N + 2 * 1 + w2];
        smem[N + 2 * 1] = l2;
    }
    __syncthreads();
    const float l = smem[N + 2 * 1];
    // out[t] = sum_i s_w[i] * v[i][t]
    const float* vcol = pv + (size_t) hkv * HD + t;
    float acc = 0.f;
    for (int i = 0; i < N; ++i) acc += s_w[i] * vcol[(size_t) i * HKV * HD];
    out[(size_t) h * HD + t] = acc * (l > 0.f ? 1.f / l : 0.f);
}

/// y[o] = sum_i W[o* n_in + i] * x[i], F32 weight (for the few F32 projections, e.g. the k/v of M3).
}  // namespace m3k
