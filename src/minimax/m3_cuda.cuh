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
__global__ void k_copy(float* __restrict__ x, const float* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = y[i];
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // y is row-major like x (blockIdx.y rows) - decode launches with grid.y=1
    x[(size_t) blockIdx.y * n + i] += y[(size_t) blockIdx.y * n + i];
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
    // y is row-major like x (the shared-expert add launches with grid.y=C); decode/expert
    // callers use grid.y=1 or per-row pointers, where base=0
    x[base + i] += s * y[base + i];
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

// ===========================================================================
// Chunked-prefill row kernels: C positions at a time (the decode path does one).
/// RoPE for C positions: q/k rows m get position t0 + m.
__global__ void k_rope_c(float* __restrict__ q, float* __restrict__ k, int H, int HKV, int HD,
                         int nrot, float t0, float base, int C) {
    const int head = blockIdx.x;
    const int m = blockIdx.y;
    float* v = (head < H ? q + (size_t) m * H * HD + (size_t) head * HD
                         : k + (size_t) m * HKV * HD + (size_t) (head - H) * HD);
    if (threadIdx.x >= nrot / 2) return;
    const int i = threadIdx.x;
    const float ang = (t0 + (float) m) * exp2f(-2.0f * (float) i / nrot * log2f(base));
    const float cos = __cosf(ang), sin = __sinf(ang);
    const float a = v[i], b = v[i + nrot / 2];
    v[i] = a * cos - b * sin;
    v[i + nrot / 2] = b * cos + a * sin;
}

/// Append C rows of K (or V) to the dense [HKV][cap][HD] cache at positions p0..p0+C-1.
__global__ void k_kv_append_c(float* __restrict__ cache, const float* __restrict__ src,
                              int HKV, int HD, int cap, int p0, int C) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    if (h >= HKV) return;
    float* dst = cache + (size_t) h * cap * HD + (size_t) (p0 + m) * HD;
    const float* s = src + ((size_t) m * HKV + h) * HD;
    for (int j = threadIdx.x; j < HD; j += blockDim.x) dst[j] = s[j];
}


// ===========================================================================
// Tiled flash attention for chunked prefill (dense-lead layers): C chunk positions attend to the
// whole P-position prefix (GPU dense cache) plus their own causal chunk positions.
//
// FA2: a block owns one head and CQ=32 query rows; it streams KV in TK=64 tiles, computes
// S = Q.K^T in shared, does the per-row online-softmax update (max/sum live in registers,
// broadcast across the row's 8 threads), and accumulates O += P.V.  K/V are read once per
// (head, tile) and reused across all 32 queries - the naive per-query loop would re-read the
// prefix C times.  96 KB dynamic smem (opt-in).
template <int CQ, int TK, int HD>
__global__ void __launch_bounds__(256)
k_fa_dense(const float* __restrict__ q,      // [C][H*HD]
           const float* __restrict__ kvk,    // [P][HKV*HD] prefix K
           const float* __restrict__ kvv,    // [P][HKV*HD] prefix V
           const float* __restrict__ ckv_k,  // [C][HKV*HD] chunk K (rope'd)
           const float* __restrict__ ckv_v,  // [C][HKV*HD] chunk V
           float* __restrict__ out,          // [C][H*HD]
           int H, int HKV, int hd, int P, int C, int cap) {
    (void) hd;   // HD is the template parameter
    // dynamic smem (90 KB - over the 48 KB static ceiling, opt-in at launch): [q tile | K tile | V tile | S tile]
    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + CQ * HD;
    float* sv = smem + CQ * HD + TK * HD;
    float* ss = smem + CQ * HD + 2 * TK * HD;
    const int h = blockIdx.x;
    const int p0 = blockIdx.y * CQ;
    const int t = threadIdx.x;
    const int hkv = h / (H / HKV);
    const float scale = rsqrtf((float) HD);
    {
        const int i = t / 8, d0 = (t % 8) * 16;
        // the last query tile may extend past C (grid is ceil(C/32)): zero those rows
        if (p0 + i < C) {
            const float* qr = q + ((size_t) (p0 + i) * H + h) * HD;
#pragma unroll
            for (int d = 0; d < 16; ++d) sq[i * HD + d0 + d] = qr[d0 + d];
        } else {
#pragma unroll
            for (int d = 0; d < 16; ++d) sq[i * HD + d0 + d] = 0.f;
        }
    }
    __syncthreads();
    const int ri = t / 8;                              // this thread's query row (within tile)
    const int rd0 = (t % 8) * 16;                      // its O dim slice
    float o[16];
#pragma unroll
    for (int d = 0; d < 16; ++d) o[d] = 0.f;
    float m = -1e30f, l = 0.f;                          // per-row online state (all 8 row-lanes equal)

    auto load_tile = [&](int t0, bool chunk) {
        // K/V tile rows 0..TK-1 at positions t0+j (chunk: ckv; else prefix kvk/kvv); zero-fill beyond bounds
        const int j = t / 4, d0 = (t % 4) * 32;
        const float* kr; const float* vr;
        if (chunk) {
            // chunk K/V are position-major [C][HKV*HD]
            kr = ckv_k + ((size_t) (t0 + j) * HKV + hkv) * HD;
            vr = ckv_v + ((size_t) (t0 + j) * HKV + hkv) * HD;
        } else {
            // the GPU dense cache is head-major [HKV][cap][HD] (k_kv_append_c layout)
            kr = kvk + (size_t) hkv * cap * HD + (size_t) (t0 + j) * HD;
            vr = kvv + (size_t) hkv * cap * HD + (size_t) (t0 + j) * HD;
        }
        const bool valid = chunk ? (t0 + j < C) : (t0 + j < P);
#pragma unroll
        for (int d = 0; d < 32; ++d) {
            sk[j * HD + d0 + d] = valid ? kr[d0 + d] : 0.f;
            sv[j * HD + d0 + d] = valid ? vr[d0 + d] : 0.f;
        }
    };
    auto tile = [&](int t0, bool chunk) {
        load_tile(t0, chunk);
        __syncthreads();
        // S[i][j] = sum_d sq[i][d] * sk[j][d], masked; 8 cols per thread
        {
            const int i = t / 8, j0 = (t % 8) * 8;
            const float* qrow = sq + i * HD;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int j = j0 + jj;
                float s = 0.f;
                const bool ok = chunk ? (t0 + j <= p0 + i) : (t0 + j < P);
                if (ok) {
                    const float* krow = sk + j * HD;
#pragma unroll
                    for (int d = 0; d < HD; ++d) s += qrow[d] * krow[d];
                    s *= scale;
                } else s = -1e30f;
                ss[i * TK + j] = s;
            }
        }
        __syncthreads();
        // online softmax row update: 8 lanes per row within one warp
        {
            const int j0 = (t % 8) * 8;
            const float* srow = ss + ri * TK + j0;
            float tm = -1e30f;
#pragma unroll
            for (int j = 0; j < 8; ++j) tm = fmaxf(tm, srow[j]);
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
            const float mn = fmaxf(m, tm);
            const float corr = (m == -1e30f) ? 0.f : expf(m - mn);
            float ls = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) ls += (srow[j] < -1e29f) ? 0.f : expf(srow[j] - mn);
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, o);
            m = mn; l = l * corr + ls;
#pragma unroll
            for (int d = 0; d < 16; ++d) o[d] *= corr;
        }
        __syncthreads();
        // O[i][d] += sum_j P[i][j] * V[j][d]; 16 dims per thread
        {
            const float* srow = ss + ri * TK;
#pragma unroll 4
            for (int j = 0; j < TK; ++j) {
                const float p = (srow[j] < -1e29f) ? 0.f : expf(srow[j] - m);
                if (p == 0.f) continue;
                const float* vrow = sv + j * HD + rd0;
#pragma unroll
                for (int d = 0; d < 16; ++d) o[d] += p * vrow[d];
            }
        }
        __syncthreads();
    };
    for (int t0 = 0; t0 < P; t0 += TK) tile(t0, false);
    for (int t0 = 0; t0 < C; t0 += TK) tile(t0, true);
    // epilogue: out = O / l
    {
        const int i = t / 8, d0 = (t % 8) * 16;
        float* orow = out + ((size_t) (p0 + i) * H + h) * HD;
        const float inv = (l > 0.f && p0 + i < C) ? 1.f / l : 0.f;
#pragma unroll
        for (int d = 0; d < 16; ++d) orow[d0 + d] = o[d] * inv;
    }
}

/// Scatter-add: x[rows[i]] += y[i] * w  (expert outputs back into the C-row residual).
__global__ void k_scatter_add(float* __restrict__ x, const int* __restrict__ rows,
                              const float* __restrict__ y, float w, int n, int NE) {
    const int i = blockIdx.x;
    if (i >= n) return;
    float* xr = x + (size_t) rows[i] * NE;
    const float* yr = y + (size_t) i * NE;
    for (int j = threadIdx.x; j < NE; j += blockDim.x) xr[j] += w * yr[j];
}

}  // namespace m3k
