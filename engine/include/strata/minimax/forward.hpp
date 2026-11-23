// include/strata/minimax/forward.hpp - one MiniMax-M3.1-Flash decoder layer, and the stack above it, in plain C++.
//
// WHAT THIS IS FOR. Strata has no kernels for this family yet, so the model has to exist somewhere as arithmetic
// before it exists as CUDA: this header is that definition. It is the thing the CUDA path is checked against, token
// for token, and it is the thing that tells us a kernel is wrong when the two disagree. It reads weights as float,
// so a caller dequantises first (strata::quant's dequant, or the file's own tensors in a test); nothing here knows
// about block formats, and nothing here knows about GPUs.
//
// THE RECIPE, and where each step comes from. Everything below follows the model's own graph
// (third_party/strata-llama/src/models/minimax-m3.cpp at the vendored pin), and the parts that are easy to guess
// wrong are marked. Dimensions in the reference's names: n_embd 6144, n_head 64, n_head_kv 4, n_embd_head 128,
// rope n_rot 64 of 128 dims (rot_pct 0.0625 of 6144 hidden), 3 dense leading layers, 57 sparse ones.
//
//   h   = x + attn(rms_norm(x, attn_norm))                                  (L303: LLM_NORM_RMS, no bias)
//   qkv = Wq h, Wk h, Wv h        Wq: n_head x 128, Wk/Wv: n_head_kv x 128  (NO output gate on q: 8192 = 64 x 128)
//   q   = rms_norm(q, attn_q_norm) per head, then partial RoPE              (L310-322: norm BEFORE rope)
//   k   = rms_norm(k, attn_k_norm) per head, then partial RoPE
//   a   = attention(q, k, v) with scale 1/sqrt(n_embd_head), causal         (L381: no softcap, no ALiBi)
//   h   = x + Wo a
//   o   = h + ffn(rms_norm(h, ffn_norm))     ffn = dense swiglu in layers < n_layer_dense_lead, else MoE
//   logits = (W_out) rms_norm(o, output_norm)                               (L593)
//
// RoPE here is the plain (non-interleaved) half-rotation over the FIRST n_rot dims of each head: dims n_rot..127
// pass through unrotated, which is what "partial RoPE" means for this family and is easy to get wrong by rotating
// the whole head.
#pragma once

#include "strata/minimax/indexer.hpp"
#include "strata/minimax/moe.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <cstddef>

namespace strata {
namespace minimax {

/// The dimensions one M3 layer needs. Everything is read from the file; nothing is hard-coded to a size.
struct LayerDims {
    int n_embd = 0;       // hidden width (6144)
    int n_head = 0;       // query heads (64)
    int n_head_kv = 0;    // KV heads (4); n_head % n_head_kv must be 0
    int head_dim = 0;     // per-head width (128)
    int n_rot = 0;        // rotated dims of each head (64); the rest of the head is not rotated
    float rope_base = 0.0f;
    float eps = 0.0f;     // attention.layer_norm_rms_epsilon (1e-6)
};

inline bool dims_ok(const LayerDims& d) {
    // n_head * head_dim is deliberately NOT required to equal n_embd: M3 projects 6144 -> 64 x 128 = 8192 for q
    // (no output gate) and 6144 -> 4 x 128 = 512 for k and v, so all three widths differ from the hidden size.
    return d.n_embd > 0 && d.n_head > 0 && d.n_head_kv > 0 && d.head_dim > 0 && d.n_rot >= 0 &&
           d.n_rot <= d.head_dim && d.n_rot % 2 == 0 && d.n_head % d.n_head_kv == 0;
}

/// RMS norm without a bias term (`LLM_NORM_RMS`), `w` may be null for the parameter-free norms.
inline void rms_norm(const float* x, const float* w, float* y, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float r = (float) (1.0 / std::sqrt(ss / (double) n + (double) eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * (w ? w[i] : 1.0f);
}

/// y = x + a, element-wise over `n`.
inline void add(float* y, const float* x, const float* a, int n) {
    for (int i = 0; i < n; ++i) y[i] = x[i] + a[i];
}

/// y = W x with W row-major `[n_out][n_in]`. The reference's projections are exactly this (a `mul_mat` on a
/// `[n_in, n_out]` GGUF tensor, which is row-major `[n_out][n_in]` once dequantised).
inline void linear(const float* w, const float* x, float* y, int n_in, int n_out) {
    for (int o = 0; o < n_out; ++o) {
        double acc = 0.0;
        const float* row = w + (size_t) o * n_in;
        for (int i = 0; i < n_in; ++i) acc += (double) row[i] * x[i];
        y[o] = (float) acc;
    }
}

/// RoPE over the first `n_rot` dims of one vector, non-interleaved (rotate i with i + n_rot/2), at `pos`.
/// Dims `n_rot..n-1` are left as they are, which is the whole point of partial RoPE.
inline void rope_partial(float* v, int n_rot, int n, float pos, float base) {
    const int half = n_rot / 2;
    for (int p = 0; p < half; ++p) {
        const float theta = pos * (float) std::pow((double) base, -2.0 * p / (double) n_rot);
        const float s = std::sin(theta), c = std::cos(theta);
        const float a = v[p], b = v[p + half];
        v[p] = a * c - b * s;
        v[p + half] = b * c + a * s;
    }
    (void) n;
}

/// Project, per-head RMS norm and partial RoPE for one token. `q` gets `n_head * head_dim` floats, `k` and `v` get
/// `n_head_kv * head_dim` each. The norm is applied to the whole head (128 dims) and RoPE only to its first `n_rot`
/// dims - in that order, as the reference writes them.
/// `qnorm_out` (optional) receives the q/k AFTER the per-head RMS norm but BEFORE the RoPE - the split point for
/// localising a divergence between the norm and the rotation.
inline void project_norm_rope(const float* h, const LayerDims& d, const float* wq, const float* wk, const float* wv,
                              const float* wq_norm, const float* wk_norm, float pos, float* q, float* k, float* v,
                              std::vector<float>* qnorm_out) {
    linear(wq, h, q, d.n_embd, d.n_head * d.head_dim);
    linear(wk, h, k, d.n_embd, d.n_head_kv * d.head_dim);
    linear(wv, h, v, d.n_embd, d.n_head_kv * d.head_dim);
    for (int i = 0; i < d.n_head; ++i) {
        float* qh = q + (size_t) i * d.head_dim;
        rms_norm(qh, wq_norm, qh, d.head_dim, d.eps);
    }
    for (int i = 0; i < d.n_head_kv; ++i) {
        float* kh = k + (size_t) i * d.head_dim;
        rms_norm(kh, wk_norm, kh, d.head_dim, d.eps);
    }
    if (qnorm_out) {
        qnorm_out->assign(q, q + (size_t) d.n_head * d.head_dim);
        qnorm_out->insert(qnorm_out->end(), k, k + (size_t) d.n_head_kv * d.head_dim);
    }
    for (int i = 0; i < d.n_head; ++i)
        rope_partial(q + (size_t) i * d.head_dim, d.n_rot, d.head_dim, pos, d.rope_base);
    for (int i = 0; i < d.n_head_kv; ++i)
        rope_partial(k + (size_t) i * d.head_dim, d.n_rot, d.head_dim, pos, d.rope_base);
}

inline void project_norm_rope(const float* h, const LayerDims& d, const float* wq, const float* wk, const float* wv,
                              const float* wq_norm, const float* wk_norm, float pos, float* q, float* k, float* v) {
    project_norm_rope(h, d, wq, wk, wv, wq_norm, wk_norm, pos, q, k, v, nullptr);
}

/// Grouped-query causal attention for one new query token. `kc`/`vc` hold `n_kv_heads * n_pos * head_dim` floats
/// (the new token's K/V are already written into column `pos`); every query head reads the KV head `i / group`.
/// The softmax is online (max, then sum) so a long cache cannot overflow, and the scale is the reference's
/// 1/sqrt(head_dim) with no softcap and no bias.
/// `ld` is the stride between positions IN HEADS (`kc` is [n_head_kv][ld][head_dim]): a cache that grew into a
/// reserved capacity has ld > n_pos, and reading it with n_pos silently shifts every KV head after the first.
inline void attn_one(const float* q, const float* kc, const float* vc, const bool* visible, const LayerDims& d,
                     int n_pos, float* out, int ld = 0) {
    if (ld <= 0) ld = n_pos;                                      // the compact case, and what the tests use
    if (!dims_ok(d) || n_pos <= 0) return;
    const int group = d.n_head / d.n_head_kv;
    const float scale = 1.0f / std::sqrt((float) d.head_dim);
    std::vector<float> acc((size_t) d.head_dim, 0.0f);           // one head's running weighted sum of values
    for (int i = 0; i < d.n_head; ++i) {
        const int kvh = i / group;
        const float* qi = q + (size_t) i * d.head_dim;
        const float* ki = kc + (size_t) kvh * ld * d.head_dim;
        const float* vi = vc + (size_t) kvh * ld * d.head_dim;
        float m = -INFINITY, l = 0.0f;
        for (int j = 0; j < d.head_dim; ++j) acc[j] = 0.0f;
        for (int p = 0; p < n_pos; ++p) {
            if (visible && !visible[p]) continue;                 // the causal mask, or a sparse selection
            const float* kp = ki + (size_t) p * d.head_dim;
            float s = 0.0f;
            for (int j = 0; j < d.head_dim; ++j) s += qi[j] * kp[j];
            s *= scale;
            const float nm = s > m ? s : m;
            const float corr = m == -INFINITY ? 0.0f : std::exp(m - nm);   // first visible key: nothing to rescale
            const float w = std::exp(s - nm);
            m = nm;
            const float* vp = vi + (size_t) p * d.head_dim;
            for (int j = 0; j < d.head_dim; ++j) acc[j] = acc[j] * corr + w * vp[j];
            l = l * corr + w;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;             // no visible key at all -> a zero row, not NaN
        for (int j = 0; j < d.head_dim; ++j) out[(size_t) i * d.head_dim + j] = acc[j] * inv;
    }
}

/// The dense feed-forward of the leading layers: gate and up are separate projections, `swiglu_oai` between them
/// (the same activation the routed experts use), then down.
inline void ffn_dense(const float* x, int n_embd, int n_ff, const float* gate, const float* up, const float* down,
                      float* out) {
    if (n_embd <= 0 || n_ff <= 0) return;
    std::vector<float> tmp((size_t) n_ff);                       // M3's dense FF is 12288 wide: no fixed cap here
    for (int o = 0; o < n_ff; ++o) {
        double a = 0.0, b = 0.0;
        const float* gr = gate + (size_t) o * n_embd;
        const float* ur = up + (size_t) o * n_embd;
        for (int i = 0; i < n_embd; ++i) { a += (double) gr[i] * x[i]; b += (double) ur[i] * x[i]; }
        tmp[o] = swiglu_oai((float) a, (float) b);
    }
    linear(down, tmp.data(), out, n_ff, n_embd);                 // down: [n_embd][n_ff]
}


/// The indexer of one sparse layer. The reference projects the hidden state to `iq` (one vector per GQA GROUP,
/// `n_groups` of them) and to a single shared `ik` per token, RMS-normalises each with its own [dim] weight, applies
/// the SAME partial RoPE as the attention, and only then takes the dot - so the cache holds normalised and rotated
/// keys, and a caller that caches raw projections scores them wrongly with no error. `n_groups == n_head_kv` is what
/// the reference asserts ("one indexer head per GQA group").
struct IndexerShape {
    int n_groups = 0;      // indexer query heads = GQA groups (4 for M3)
    int dim = 0;           // indexer head width (128)
    int blk = 0;           // positions per block (128)
    int topk = 0;          // blocks per query (16)
    int local = 0;         // recent blocks forced in (1)
    float rope_base = 0.0f;
    float eps = 0.0f;
};

/// One indexer vector, prepared for the cache or for a query: RMS norm with `w`, then partial RoPE at `pos`.
/// `n_rot` is the same rotated width the attention uses (64 of 128), because the reference passes the same n_rot.
inline void indexer_prep(const float* src, const float* w, int dim, int n_rot, float pos, float base, float eps,
                         float* dst) {
    rms_norm(src, w, dst, dim, eps);
    rope_partial(dst, n_rot, dim, pos, base);
}

/// Which positions one query token may read, through the indexer. `iq` is `n_groups` prepared query vectors,
/// `keys` is the prepared key cache (`n_pos` x dim, one shared key per position), `mask` is 0.0 on written positions
/// and -INFINITY on holes. `sel` receives `n_groups` selections and `visible` `n_pos` flags - the UNION over groups,
/// because a KV head in group g reads the blocks its own group selected and the caller passes per-group visibility
/// if it wants it exactly (this flat form is what a CPU reference and a mask cache use).
inline void msa_visible(const float* iq, const float* keys, const float* mask, const IndexerShape& s, int n_pos,
                        int q_pos, BlockSel* sel, bool* visible) {
    for (int i = 0; i < n_pos; ++i) visible[i] = false;
    if (!iq || !keys || s.n_groups <= 0 || s.blk <= 0 || s.topk <= 0) return;
    const int n_blk = (n_pos + s.blk - 1) / s.blk;
    std::vector<float> scores((size_t) s.blk > 0 ? n_blk : 1);
    for (int g = 0; g < s.n_groups; ++g) {
        block_scores(iq + (size_t) g * s.dim, keys, mask, n_pos, s.dim, s.blk, q_pos, s.local, scores.data());
        sel[g] = select_blocks(scores.data(), n_blk, s.topk);
        for (int j = 0; j < sel[g].n; ++j) {
            const int b = sel[g].blocks[j];
            for (int p = b * s.blk; p < (b + 1) * s.blk && p < n_pos; ++p) {
                // a hole scores -INFINITY so it never ranks a block in, but it can sit INSIDE a block that was
                // chosen for another position - and it must not be read. The graph masks it; this reference does too.
                if (mask && mask[p] != 0.0f) continue;
                visible[p] = true;
            }
        }
    }
}

/// One sparse layer's routed experts plus its shared expert. Every weight matrix is dequantised float, row-major,
/// and the expert stacks are the file's own order: row `e * n_ff + j` of `gate` is expert e's row j.
struct MoEWeights {
    const float* router = nullptr;   // [n_expert][n_embd]   ffn_gate_inp.weight
    const float* bias = nullptr;     // [n_expert]           exp_probs_b.bias
    const float* gate = nullptr;     // [n_expert*n_ff][n_embd]
    const float* up = nullptr;       // [n_expert*n_ff][n_embd]
    const float* down = nullptr;     // [n_expert*n_embd][n_ff]
    const float* sh_gate = nullptr;  // shared expert, may all be null
    const float* sh_up = nullptr;
    const float* sh_down = nullptr;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff = 0;
    int n_embd = 0;
    float scale = 1.0f;              // expert_weights_scale (2.0); 0 and 1 both mean unscaled
    bool norm = true;                // expert_weights_norm
};

/// One expert's row: down(swiglu_oai(gate x, up x)). `h` is scratch of `n_ff` floats, `out` gets `n_embd`.
inline void expert_row(const float* x, int n_embd, int n_ff, const float* gate, const float* up, const float* down,
                       float* h, float* out) {
    for (int j = 0; j < n_ff; ++j) {
        double a = 0.0, b = 0.0;
        const float* gr = gate + (size_t) j * n_embd;
        const float* ur = up + (size_t) j * n_embd;
        for (int i = 0; i < n_embd; ++i) { a += (double) gr[i] * x[i]; b += (double) ur[i] * x[i]; }
        h[j] = swiglu_oai((float) a, (float) b);
    }
    linear(down, h, out, n_ff, n_embd);
}

/// The routed experts and the shared expert in one pass. `route_out` may be null; it receives the selection, which
/// is the part worth looking at when an answer is wrong.
inline void moe_ffn(const float* x, const MoEWeights& w, Route* route_out, float* out) {
    if (w.n_expert <= 0 || w.n_embd <= 0 || w.n_ff <= 0) return;
    std::vector<float> logits((size_t) w.n_expert);
    linear(w.router, x, logits.data(), w.n_embd, w.n_expert);
    const Route r = route(logits.data(), w.bias, w.n_expert, w.n_expert_used, w.norm, w.scale);
    if (route_out) *route_out = r;
    std::vector<float> h((size_t) w.n_ff);
    // One buffer per selected expert: combine() reads all the rows at the end, so a single scratch row would hand
    // every weight the LAST expert's output - four copies of one expert and a wrong answer with no error.
    std::vector<std::vector<float>> rows((size_t) r.n, std::vector<float>((size_t) w.n_embd));
    std::vector<const float*> rowp((size_t) r.n);
    for (int s = 0; s < r.n; ++s) {
        const int e = r.experts[s];
        expert_row(x, w.n_embd, w.n_ff, w.gate + (size_t) e * w.n_ff * w.n_embd, w.up + (size_t) e * w.n_ff * w.n_embd,
                   w.down + (size_t) e * w.n_embd * w.n_ff, h.data(), rows[s].data());
        rowp[s] = rows[s].data();
    }
    std::vector<float> shared;
    if (w.sh_gate && w.sh_up && w.sh_down) {
        shared.resize((size_t) w.n_embd);
        expert_row(x, w.n_embd, w.n_ff, w.sh_gate, w.sh_up, w.sh_down, h.data(), shared.data());
    }
    combine(r, rowp.data(), shared.empty() ? nullptr : shared.data(), w.n_embd, out);
}

}  // namespace minimax
}  // namespace strata
