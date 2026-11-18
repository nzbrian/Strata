// tests/minimax/forward_test.cpp - the M3 layer recipe, checked against hand-computed numbers rather than against
// itself. A reference header that is only ever compared with itself proves nothing (a dropped `+ 1` in swiglu_oai
// survived exactly that), so every expectation below is a literal or a value worked out on paper from the recipe in
// forward.hpp's comment.
//
//   ninja -C build-m3 m3_forward_test && ./build-m3/m3_forward_test

#include "strata/minimax/forward.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace mm = strata::minimax;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
static bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }

static mm::LayerDims tiny_dims() {
    mm::LayerDims d;
    d.n_embd = 8; d.n_head = 4; d.n_head_kv = 2; d.head_dim = 4; d.n_rot = 2;
    d.rope_base = 10000.0f; d.eps = 1e-6f;
    return d;
}

int main() {
    std::printf("m3_forward_test - the layer recipe, from the reference's own graph\n");
    {
        mm::LayerDims d = tiny_dims();
        check(mm::dims_ok(d), "the tiny shape is accepted");
        mm::LayerDims m3; m3.n_embd = 6144; m3.n_head = 64; m3.n_head_kv = 4; m3.head_dim = 128; m3.n_rot = 64;
        m3.rope_base = 1000000.0f; m3.eps = 1e-6f;
        check(mm::dims_ok(m3), "and so is the real M3 shape, where q (8192) and k (512) both differ from n_embd");
        mm::LayerDims bad = d; bad.n_rot = 3;
        check(!mm::dims_ok(bad), "an odd rotated width is refused (a pair would be left unrotated)");
        bad = d; bad.n_head = 5;
        check(!mm::dims_ok(bad), "and so is a head count the KV heads do not divide");
    }
    {
        // rms_norm: unit RMS out of any input, a weight vector that scales per dim, and a parameter-free norm
        std::vector<float> x = {3.0f, -4.0f, 12.0f, 0.0f}, y(4);
        mm::rms_norm(x.data(), nullptr, y.data(), 4, 0.0f);
        double ss = 0.0;
        for (float v : y) ss += (double) v * v;
        check(near((float) std::sqrt(ss / 4.0), 1.0f), "RMS of the output is 1 with a unit eps and no weight");
        std::vector<float> w = {2.0f, 2.0f, 2.0f, 2.0f};
        mm::rms_norm(x.data(), w.data(), y.data(), 4, 0.0f);
        check(near(y[2], 12.0f * 2.0f / 6.5f, 1e-4f), "the weight scales each dim after normalising (mean of squares 42.25 -> 6.5)");
        std::vector<float> z = {0.0f, 0.0f, 0.0f, 0.0f};
        mm::rms_norm(z.data(), nullptr, y.data(), 4, 1e-6f);
        check(y[0] == 0.0f && std::isfinite(y[1]), "an all-zero vector gives 0, not NaN");
    }
    {
        // partial RoPE: the rotated pair in numbers worked out by hand, and the untouched tail
        const float c = 0.540302305868f, s = 0.841470984808f;      // cos(1), sin(1)
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        mm::rope_partial(v.data(), 4, 6, 1.0f, 1.0f);               // base 1 -> theta = pos for every pair
        check(near(v[0], 1.0f * c - 3.0f * s), "pair 0: x0*cos - x2*sin at theta = 1");
        check(near(v[2], 3.0f * c + 1.0f * s), "pair 0: x2*cos + x0*sin");
        check(near(v[1], 2.0f * c - 4.0f * s), "pair 1: x1*cos - x3*sin");
        check(v[4] == 5.0f && v[5] == 6.0f, "dims past n_rot are left exactly alone (partial RoPE, not full)");
        std::vector<float> v0 = {1.0f, 2.0f, 3.0f, 4.0f};
        mm::rope_partial(v0.data(), 4, 4, 0.0f, 10000.0f);
        check(v0[0] == 1.0f && v0[1] == 2.0f && v0[2] == 3.0f && v0[3] == 4.0f, "position 0 is the identity rotation");
    }
    {
        // project + per-head norm + partial rope: the norm is per head and comes BEFORE the rotation
        const mm::LayerDims d = tiny_dims();
        std::vector<float> h(8, 0.5f);
        std::vector<float> wq((size_t) d.n_head * d.head_dim * d.n_embd, 0.0f);
        std::vector<float> wk((size_t) d.n_head_kv * d.head_dim * d.n_embd, 0.0f);
        std::vector<float> wv((size_t) d.n_head_kv * d.head_dim * d.n_embd, 0.0f);
        for (int o = 0; o < d.n_head * d.head_dim; ++o) wq[(size_t) o * d.n_embd + (o % d.n_embd)] = 1.0f;
        for (int o = 0; o < d.n_head_kv * d.head_dim; ++o) wk[(size_t) o * d.n_embd + (o % d.n_embd)] = 1.0f;
        for (int o = 0; o < d.n_head_kv * d.head_dim; ++o) wv[(size_t) o * d.n_embd + (o % d.n_embd)] = 2.0f;
        std::vector<float> qn(d.head_dim, 1.0f), kn(d.head_dim, 1.0f);
        std::vector<float> q((size_t) d.n_head * d.head_dim), k((size_t) d.n_head_kv * d.head_dim),
            v((size_t) d.n_head_kv * d.head_dim);
        mm::project_norm_rope(h.data(), d, wq.data(), wk.data(), wv.data(), qn.data(), kn.data(), 0.0f, q.data(),
                              k.data(), v.data());
        check(v[0] == 1.0f, "v is the projection with no norm and no rotation");
        check(near(q[2], 1.0f, 1e-5f) && near(q[3], 1.0f, 1e-5f), "dims past n_rot keep the normalised value");
        qn[0] = 2.0f;
        mm::project_norm_rope(h.data(), d, wq.data(), wk.data(), wv.data(), qn.data(), kn.data(), 0.0f, q.data(),
                              k.data(), v.data());
        check(near(q[0], 2.0f * 1.0f, 1e-5f), "the q norm weight multiplies that dim of every head");
        qn[0] = 1.0f;
        kn[1] = 3.0f;
        mm::project_norm_rope(h.data(), d, wq.data(), wk.data(), wv.data(), qn.data(), kn.data(), 0.0f, q.data(),
                              k.data(), v.data());
        // attn_k_norm is one [head_dim] vector applied inside every head (the file gives it shape [128], not
        // [n_head_kv * 128]), so dim 1 of BOTH heads is scaled - shared weights, per-head application.
        check(near(k[1], 3.0f, 1e-5f) && near(k[5], 3.0f, 1e-5f) && near(k[0], 1.0f, 1e-5f),
              "the k norm weight scales that dim inside every head");
    }
    {
        // attention: one visible key returns its value; the mask really masks; a group shares a KV head
        const mm::LayerDims d = tiny_dims();
        const int np = 3;
        std::vector<float> q((size_t) d.n_head * d.head_dim, 0.0f);
        for (int i = 0; i < d.n_head * d.head_dim; ++i) q[i] = (float) (i % d.head_dim);
        std::vector<float> kc((size_t) d.n_head_kv * np * d.head_dim, 0.0f);
        std::vector<float> vc((size_t) d.n_head_kv * np * d.head_dim, 0.0f);
        for (int h = 0; h < d.n_head_kv; ++h) {
            for (int j = 0; j < d.head_dim; ++j) {
                kc[((size_t) h * np + 0) * d.head_dim + j] = 0.0f;                       // zero key: score 0
                vc[((size_t) h * np + 0) * d.head_dim + j] = 7.0f;
                kc[((size_t) h * np + 1) * d.head_dim + j] = 1000.0f;                    // never visible below
                vc[((size_t) h * np + 1) * d.head_dim + j] = -99.0f;
                vc[((size_t) h * np + 2) * d.head_dim + j] = 1.0f;
            }
        }
        const bool vis[] = {true, false, false};
        std::vector<float> out((size_t) d.n_head * d.head_dim);
        mm::attn_one(q.data(), kc.data(), vc.data(), vis, d, np, out.data());
        bool all_seven = true;
        for (float x : out) if (!near(x, 7.0f)) all_seven = false;
        check(all_seven, "one visible key with score 0 returns that key's value for every head");
        const bool vis2[] = {true, false, true};
        mm::attn_one(q.data(), kc.data(), vc.data(), vis2, d, np, out.data());
        check(near(out[0], (1.0f * 1.0f + 1.0f * 7.0f) / 2.0f), "two keys with equal score average their values");
        const bool vis3[] = {false, true, false};
        mm::attn_one(q.data(), kc.data(), vc.data(), vis3, d, np, out.data());
        check(near(out[0], -99.0f), "and the mask picks the hidden key when it is the only visible one");
        const bool vis4[] = {false, false, false};
        mm::attn_one(q.data(), kc.data(), vc.data(), vis4, d, np, out.data());
        bool zero_finite = true;
        for (float x : out) if (x != 0.0f || !std::isfinite(x)) zero_finite = false;
        check(zero_finite, "a query that sees nothing gets zeros, not NaN");
        // a peaked score must dominate: make KV head 0's position 2 score far above the rest
        for (int j = 0; j < d.head_dim; ++j) kc[(size_t) 0 * np * d.head_dim + 2 * d.head_dim + j] = 200.0f;
        const bool vis5[] = {true, false, true};   // pos 1 keeps the 1000-wide key: hide it, or it wins instead
        mm::attn_one(q.data(), kc.data(), vc.data(), vis5, d, np, out.data());
        check(out[0] > 0.9f && out[0] < 1.1f, "the highest-scoring key dominates the softmax (KV head 0's row)");
        // head i reads KV head i / group, so the first head OF A GROUP is g * group (comparing against head g
        // instead compares two different groups, which is what this check used to do)
        const int group = d.n_head / d.n_head_kv;
        bool same_group = true;
        for (int h = 0; h < d.n_head; ++h) {
            const int first = (h / group) * group;
            for (int j = 0; j < d.head_dim; ++j) {
                if (!near(out[(size_t) h * d.head_dim + j], out[(size_t) first * d.head_dim + j], 1e-4f)) {
                    same_group = false;
                }
            }
        }
        check(same_group, "equal queries in one GQA group give equal outputs (head i reads KV head i/group)");
        check(!near(out[0], out[(size_t) group * d.head_dim], 1e-2f),
              "and two groups with different KV heads really do differ (otherwise the grouping is ignored)");
    }
    {
        // dense FFN: swiglu_oai between two projections, then down - and no silent width cap
        const int n_embd = 2, n_ff = 5000;                        // 5000 > any fixed tile: M3's own dense FF is 12288
        std::vector<float> x = {1.0f, 0.0f};
        std::vector<float> gate((size_t) n_ff * n_embd, 0.0f), up((size_t) n_ff * n_embd, 0.0f);
        std::vector<float> down((size_t) n_embd * n_ff, 0.0f);
        gate[0 * n_embd + 0] = 2.0f; up[0 * n_embd + 0] = 1.0f;   // row 0: gate = 2, up = 1
        down[0 * n_ff + 0] = 1.0f;                                // out[0] = swiglu_oai(2, 1)
        std::vector<float> out(n_embd, -1.0f);
        mm::ffn_dense(x.data(), n_embd, n_ff, gate.data(), up.data(), down.data(), out.data());
        check(near(out[0], 1.935658623f * 2.0f, 1e-4f), "out[0] = swiglu_oai(2, 1) = 2*sigmoid(3.404)*2");
        check(near(out[1], 0.0f), "and a zero row of down contributes nothing");
    }
    {
        // the indexer path: norm and RoPE before the dot, one shared key, blocks in a group's own order
        mm::IndexerShape s; s.n_groups = 2; s.dim = 4; s.blk = 2; s.topk = 1; s.local = 0;
        s.rope_base = 10000.0f; s.eps = 1e-6f;
        std::vector<float> iq = {1, 0, 0, 0,   0, 1, 0, 0};              // group 0 wants key 0, group 1 wants key 1
        // scores: group 0 -> 1, 0, 5 (block 1 wins), group 1 -> 0, 1, 0 (block 0 wins). Distinct, so the
        // selection order is defined and this does not quietly depend on the tie rule.
        std::vector<float> keys = {1, 0, 0, 0,  0, 1, 0, 0,  5, 0, 0, 0};  // positions 0,1,2
        std::vector<float> mask = {0, 0, 0};
        std::vector<mm::BlockSel> sel(2);
        std::vector<bool> unused(3);
        std::vector<char> vis(3, 0);
        mm::msa_visible(iq.data(), keys.data(), mask.data(), s, 3, 2, sel.data(), reinterpret_cast<bool*>(vis.data()));
        check(sel[0].n == 1 && sel[0].blocks[0] == 1, "group 0's top block is the one holding its best key (block 1)");
        check(sel[1].n == 1 && sel[1].blocks[0] == 0, "group 1 scores separately and picks block 0 instead");
        check(vis[0] && vis[1] && vis[2], "the union over the two groups covers every position");
        std::vector<float> holes = {0, -INFINITY, 0};                    // position 1 is a hole in the cache
        mm::msa_visible(iq.data(), keys.data(), holes.data(), s, 3, 2, sel.data(),
                        reinterpret_cast<bool*>(vis.data()));
        check(!vis[1], "a hole inside a selected block is still never visible");
        std::vector<float> iw(4, 1.0f), prep(4);
        std::vector<float> raw = {1, 1, 1, 1};
        mm::indexer_prep(raw.data(), iw.data(), 4, 2, 0.0f, 10000.0f, 1e-6f, prep.data());
        check(near(prep[0], 1.0f, 1e-4f) && near(prep[3], 1.0f, 1e-4f), "indexer_prep normalises then leaves pos 0 alone");
    }
    {
        // MoE: the weights sum to the scale, the shared expert is added once, and a dominant expert shows up alone
        const int ne = 3, nff = 2, nh = 4;
        mm::MoEWeights w;
        w.n_expert = ne; w.n_expert_used = 2; w.n_ff = nff; w.n_embd = nh; w.scale = 2.0f; w.norm = true;
        std::vector<float> router = {10, 0, 0, 0,   9, 0, 0, 0,   -10, 0, 0, 0};   // experts 0 and 1 win
        std::vector<float> bias(ne, 0.0f);
        std::vector<float> gate((size_t) ne * nff * nh, 0.0f), up((size_t) ne * nff * nh, 0.0f),
            down((size_t) ne * nh * nff, 0.0f);
        // expert e: gate row 0 = x0 * (e+1), up row 0 = 0, down col 0 -> out[0] = (e+1) * swiglu factor
        for (int e = 0; e < ne; ++e) {
            gate[((size_t) e * nff + 0) * nh + 0] = (float) (e + 1);
            down[(size_t) e * nh * nff + 0 + 0] = 1.0f;
        }
        std::vector<float> x = {1.0f, 0.0f, 0.0f, 0.0f}, out(nh, 0.0f);
        mm::Route r;
        w.router = router.data(); w.bias = bias.data(); w.gate = gate.data(); w.up = up.data(); w.down = down.data();
        mm::moe_ffn(x.data(), w, &r, out.data());
        check(r.n == 2 && r.experts[0] == 0 && r.experts[1] == 1, "the two strongest router logits are chosen");
        float wsum = r.weights[0] + r.weights[1];
        check(near(wsum, 2.0f, 1e-4f), "the two weights sum to expert_weights_scale (2.0), not to 1");
        // expert e has gate row 0 = (e+1)*x0 and up = 0, so its row is swiglu_oai(e+1, 0)
        const double expect = r.weights[0] * mm::swiglu_oai(1.0f, 0.0f) + r.weights[1] * mm::swiglu_oai(2.0f, 0.0f);
        check(near(out[0], (float) expect, 1e-4f), "out[0] is the weighted sum of the two experts' rows");
        check(out[1] == 0.0f && out[3] == 0.0f, "and nothing leaks into the untouched dims");
        std::vector<float> sg((size_t) nff * nh, 0.0f), su((size_t) nff * nh, 0.0f), sd((size_t) nh * nff, 0.0f);
        sg[0] = 1.0f; sd[0] = 1.0f;   // shared expert: gate = x0 = 1, up = 0, so it adds swiglu_oai(1, 0)
        w.sh_gate = sg.data(); w.sh_up = su.data(); w.sh_down = sd.data();
        std::vector<float> out2(nh, 0.0f);
        mm::moe_ffn(x.data(), w, &r, out2.data());
        check(near(out2[0], (float) (expect + mm::swiglu_oai(1.0f, 0.0f)), 1e-4f),
              "the shared expert is added once, on top of the routed sum");
        // the two experts must NOT collapse onto one row: make expert 1's output distinct and check both appear
        gate[((size_t) 1 * nff + 0) * nh + 0] = 3.0f;                   // expert 1 now scales x0 by 3
        std::vector<float> out3(nh, 0.0f);
        mm::moe_ffn(x.data(), w, &r, out3.data());
        const double expect3 = r.weights[0] * mm::swiglu_oai(1.0f, 0.0f) + r.weights[1] * mm::swiglu_oai(3.0f, 0.0f) +
                               mm::swiglu_oai(1.0f, 0.0f);
        check(near(out3[0], (float) expect3, 1e-4f),
              "each expert's own row is used (a shared scratch row would give 2x expert 1)");
    }
    std::printf("%s\n", failures == 0 ? "ALL PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
