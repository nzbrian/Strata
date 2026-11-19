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

/// GQA causal attention, one query row (H heads) against npos KV positions, out = H x HD. cap is the KV row stride
/// (the cache capacity). Matches strata::minimax::attn_one.
__global__ void k_attn(const float* __restrict__ q, const float* __restrict__ kv, const float* __restrict__ vv,
                       float* __restrict__ out, int H, int HKV, int HD, int npos, int cap, int qpos) {
    const int h = blockIdx.x;
    if (h >= H) return;
    const int hkv = h / (H / HKV);
    // scores for head h over the visible positions 0..qpos
    const float* qh = q + (size_t) h * HD;
    const float* kh = kv + (size_t) hkv * cap * HD;
    __shared__ float s[256];
    const int nvis = qpos + 1;
    // compute the scores into shared (one thread per position)
    for (int p = threadIdx.x; p < nvis; p += blockDim.x) {
        float d = 0.0f;
        const float* kp = kh + (size_t) p * HD;
        for (int j = 0; j < HD; ++j) d += qh[j] * kp[j];
        s[p] = d * rsqrtf((float) HD);
    }
    __syncthreads();
    // softmax over the visible positions
    float mx = -1e30f;
    for (int p = 0; p < nvis; ++p) mx = fmaxf(mx, s[p]);
    float sum = 0.0f;
    for (int p = 0; p < nvis; ++p) { s[p] = __expf(s[p] - mx); sum += s[p]; }
    for (int p = 0; p < nvis; ++p) s[p] /= sum;
    __syncthreads();
    // weighted sum of the V rows
    const float* vh = vv + (size_t) hkv * cap * HD;
    if (threadIdx.x < HD) {
        float acc = 0.0f;
        for (int p = 0; p < nvis; ++p) acc += s[p] * vh[(size_t) p * HD + threadIdx.x];
        out[(size_t) h * HD + threadIdx.x] = acc;
    }
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
/// (H heads). `vis` is npos bytes, 1 on a position to read. Matches strata::minimax::attn_one with a sparse mask.
__global__ void k_msa_attn(const float* __restrict__ q, const float* __restrict__ kv, const float* __restrict__ vv,
                           const unsigned char* __restrict__ vis, float* __restrict__ out,
                           int H, int HKV, int HD, int npos, int cap) {
    const int h = blockIdx.x;
    if (h >= H) return;
    const int hkv = h / (H / HKV);
    const float* qh = q + (size_t) h * HD;
    const float* kh = kv + (size_t) hkv * cap * HD;
    const float* vh = vv + (size_t) hkv * cap * HD;
    __shared__ int vispos[2048];   // the visible position indices (topk*blk = 16*128)
    __shared__ float s[2048];
    __shared__ int nvis;
    if (threadIdx.x == 0) {
        int c = 0;
        for (int p = 0; p < npos; ++p) if (vis[p]) vispos[c++] = p;
        nvis = c;
    }
    __syncthreads();
    const int nv = nvis;
    for (int p = threadIdx.x; p < nv; p += blockDim.x) {
        float d = 0.0f;
        const float* kp = kh + (size_t) vispos[p] * HD;
        for (int j = 0; j < HD; ++j) d += qh[j] * kp[j];
        s[p] = d * rsqrtf((float) HD);
    }
    __syncthreads();
    float mx = -1e30f;
    for (int p = 0; p < nv; ++p) mx = fmaxf(mx, s[p]);
    float sum = 0.0f;
    for (int p = 0; p < nv; ++p) { s[p] = __expf(s[p] - mx); sum += s[p]; }
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    __syncthreads();
    if (threadIdx.x < HD) {
        float acc = 0.0f;
        for (int p = 0; p < nv; ++p) acc += s[p] * inv * vh[(size_t) vispos[p] * HD + threadIdx.x];
        out[(size_t) h * HD + threadIdx.x] = acc;
    }
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

}  // namespace m3k
