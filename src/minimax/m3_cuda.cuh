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

}  // namespace m3k
