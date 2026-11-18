// tools/m3ref/m3_native.cpp - MiniMax-M3.1-Flash's forward pass in Strata's own C++, on real weights, on CPU.
//
// WHY THIS EXISTS. The engine still refuses this architecture, so before a CUDA kernel exists there has to be a
// Strata-side answer to "what does this model compute". This is it: GGUF in through the artifact reader, quantised
// rows through strata::quant's dequant, arithmetic from include/strata/minimax. No libllama, no llama.cpp graph, no
// tokenizer - token ids come in on the command line (tools/strata_tokenizer.py makes them, and its test round-trips
// the real 200,064-token vocabulary). With all layers on, the top-1 id it prints is the model's next token, which is
// what the CUDA path has to reproduce.
//
// MEMORY. The routed experts are 95.89 GiB of the file; dequantised to float they would be ~450 GiB, so they are
// dequantised one expert slot at a time into reused buffers. The rest is loaded per layer and freed before the next
// layer, which keeps the peak near one layer's weights plus the embedding and the output head.
//
// Build: ninja -C build-m3 m3_native            Run: ./build-m3/m3_native --model <shard-1.gguf> --tokens 1,2,3 [--layers N] [--report]

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/geometry.hpp"
#include "strata/minimax/forward.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm = strata::minimax;
namespace core = strata::core;
namespace Q = strata::quant;

namespace {

double secs(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

/// One tensor's elements as floats. F32 needs no dequantising - `can_dequant` is false for it on purpose, because
/// there is nothing to decode - so it is a copy; everything else goes through the Strata quant layer.
bool to_floats(Q::Type ty, const uint8_t* src, uint64_t n, std::vector<float>& out, std::string& err) {
    out.assign((size_t) n, 0.0f);
    if (ty == Q::Type::F32) {
        std::memcpy(out.data(), src, (size_t) n * sizeof(float));
        return true;
    }
    if (!Q::can_dequant(ty)) { err = "no float path for type " + std::to_string((int) ty); return false; }
    if (!Q::dequant_row(ty, src, out.data(), (int64_t) n)) { err = "dequant failed for type " + std::to_string((int) ty); return false; }
    return true;
}

/// The weights this pass needs, read from the file as float rows and kept only while the layer using them runs.
class Model {
  public:
    bool open(const std::string& any_shard, std::string& err) {
        try {
            file_ = std::make_unique<strata::GgufModel>(strata::GgufModel::open(any_shard));
        } catch (const std::exception& e) {
            err = std::string("cannot open: ") + e.what();
            return false;
        }
        const strata::GgufFile& meta = file_->meta();
        if (strata::model_family(meta) != strata::ModelFamily::MiniMaxM3) { err = "this runner takes a minimax-m3 file"; return false; }
        const std::string aerr = strata::check_architecture(meta);
        if (!aerr.empty()) { err = aerr; return false; }
        int64_t top_k = 0;
        if (!core::geometry_from_gguf(meta, geom, top_k, err)) return false;

        dims.n_embd = (int) geom.n_embd;
        dims.n_head = (int) geom.n_head;
        dims.n_head_kv = (int) geom.n_head_kv;
        dims.head_dim = (int) geom.head_dim;
        dims.n_rot = (int) geom.n_rot;
        dims.eps = (float) geom.rms_eps;
        const strata::MetaValue* fb = meta.get("minimax-m3.rope.freq_base");
        dims.rope_base = fb && fb->is_num() ? (float) fb->num() : 1000000.0f;
        idx.n_groups = (int) geom.idx_q_heads;
        idx.dim = (int) geom.idx_key_dim;
        idx.blk = (int) geom.idx_block;
        idx.topk = (int) geom.idx_top_k;
        idx.local = (int) geom.idx_local_blocks;
        idx.rope_base = dims.rope_base;
        idx.eps = dims.eps;
        return true;
    }

    /// Free everything but the globals, before a layer's weights are read.
    void begin_layer() {
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (it->first.rfind("blk.", 0) == 0) it = cache_.erase(it);
            else ++it;
        }
    }

    /// A whole tensor as float rows. GGUF's dim 0 varies fastest, so this is row-major [ne1][ne0], which is the
    /// (n_out, n_in) form strata::minimax::linear takes. Null (with `err` set) if the file or the type says no.
    const std::vector<float>* get(const std::string& name, std::string& err) {
        const auto hit = cache_.find(name);
        if (hit != cache_.end()) return hit->second.empty() ? nullptr : &hit->second;
        std::vector<float> v;
        size_t si = 0;
        const strata::TensorInfo* t = file_->find(name, &si);
        if (!t) { err = "missing tensor " + name; cache_.emplace(name, std::vector<float>{}); return nullptr; }
        if (!to_floats((Q::Type) t->type, file_->shard(si).tensor_data(*t), t->elements(), v, err)) {
            err = name + ": " + err;
            cache_.emplace(name, std::vector<float>{});
            return nullptr;
        }
        return &cache_.emplace(name, std::move(v)).first->second;
    }

    /// One expert of one layer's stacked tensors. The stack is the slowest dimension, so an expert's bytes are
    /// contiguous and ne0 * ne1 is a whole number of blocks, so one slot is one dequant call.
    bool slot(int layer, int expert, const char* what, std::vector<float>& out, std::string& err) {
        const std::string name = "blk." + std::to_string(layer) + "." + what;
        size_t si = 0;
        const strata::TensorInfo* t = file_->find(name, &si);
        if (!t || t->shape.size() < 3) { err = "missing stacked tensor " + name; return false; }
        if (expert < 0 || (uint64_t) expert >= t->shape[2]) { err = "expert index out of range in " + name; return false; }
        const Q::Type ty = (Q::Type) t->type;
        const uint64_t per = t->shape[0] * t->shape[1];
        const size_t total = Q::row_bytes((int) ty, (int64_t) t->elements());
        if (total == 0 || t->shape[2] == 0 || total % t->shape[2] != 0) { err = name + ": unequal expert strides"; return false; }
        const uint8_t* src = file_->shard(si).tensor_data(*t) + (size_t) expert * (total / t->shape[2]);
        if (!to_floats(ty, src, per, out, err)) { err = name + ": " + err; return false; }
        return true;
    }

    const std::vector<float>* ne_of(const std::string& name, size_t& ne0, size_t& ne1) {
        static std::string ignore;
        const std::vector<float>* v = get(name, ignore);
        const strata::TensorInfo* t = file_->find(name);
        if (!v || !t || t->shape.size() < 2) return nullptr;
        ne0 = (size_t) t->shape[0];
        ne1 = (size_t) t->shape[1];
        return v;
    }

    mm::LayerDims dims;
    mm::IndexerShape idx;
    core::ModelGeometry geom;

  private:
    std::unique_ptr<strata::GgufModel> file_;
    std::unordered_map<std::string, std::vector<float>> cache_;
};

/// The KV cache of one layer, as float rows per KV head, positions appended in order.
struct KvCache {
    std::vector<float> k, v;
    int n_pos = 0, cap = 0;
    void reserve(int n_kv, int dim, int capacity) {
        cap = capacity;
        k.assign((size_t) n_kv * cap * dim, 0.0f);
        v.assign(k.size(), 0.0f);
        n_pos = 0;
    }
};

struct Report {
    int tokens = 0;
    double min_gap = 1e30, sum_w = 0.0;
    int tight = 0, checked = 0, nan_hits = 0;
    std::vector<int> expert_hits;
};

}  // namespace

int main(int argc, char** argv) {
    std::string model;
    std::vector<int> tokens;
    int layers_wanted = 0, top = 5;
    bool report = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](std::string& d) { return i + 1 < argc ? (d = argv[++i], true) : false; };
        std::string v;
        if (k == "--model") { if (!next(model)) return 2; }
        else if (k == "--tokens") {
            if (!next(v)) return 2;
            size_t p = 0;
            while (p <= v.size()) {
                const size_t c = v.find(',', p);
                const std::string s = v.substr(p, c == std::string::npos ? std::string::npos : c - p);
                if (!s.empty()) tokens.push_back(std::atoi(s.c_str()));
                if (c == std::string::npos) break;
                p = c + 1;
            }
        } else if (k == "--layers") { if (!next(v)) return 2; layers_wanted = std::atoi(v.c_str()); }
        else if (k == "--top") { if (!next(v)) return 2; top = std::atoi(v.c_str()); }
        else if (k == "--report") report = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || tokens.empty()) {
        std::fprintf(stderr, "usage: m3_native --model <shard-1.gguf> --tokens 1,2,3 [--layers N] [--top K] [--report]\n");
        return 2;
    }

    std::string err;
    Model m;
    if (!m.open(model, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int T = (int) tokens.size();
    const int nL = layers_wanted > 0 && layers_wanted < (int) m.geom.n_layers ? layers_wanted : (int) m.geom.n_layers;
    const int NE = m.dims.n_embd, H = m.dims.n_head, HKV = m.dims.n_head_kv, HD = m.dims.head_dim;
    std::printf("geometry      %d layers of %lld, hidden %d, heads %d/%d x %d, rot %d, experts %lld top-%lld, dense lead %d\n",
                nL, (long long) m.geom.n_layers, NE, H, HKV, HD, m.dims.n_rot, (long long) m.geom.n_expert,
                (long long) m.geom.n_expert_used, (int) m.geom.n_dense_lead);
    std::printf("indexer       %d groups x %d dims, block %d, top-k %d, local %d\n", m.idx.n_groups, m.idx.dim, m.idx.blk,
                m.idx.topk, m.idx.local);

    // x[t] = the token's embedding row
    std::vector<std::vector<float>> x((size_t) T);
    {
        size_t ne0 = 0, ne1 = 0;
        const std::vector<float>* e = m.ne_of("token_embd.weight", ne0, ne1);
        if (!e) { std::fprintf(stderr, "embedding: %s\n", "token_embd.weight unavailable"); return 1; }
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || (size_t) tokens[t] >= ne1) { std::fprintf(stderr, "token id %d out of vocab %zu\n", tokens[t], ne1); return 1; }
            x[t].assign(e->begin() + (long) ((size_t) tokens[t] * ne0), e->begin() + (long) ((size_t) (tokens[t] + 1) * ne0));
        }
    }
    std::vector<KvCache> kv((size_t) nL);
    std::vector<float> ik_cache;                                       // prepared indexer keys, per layer
    Report rep;
    rep.expert_hits.assign((size_t) m.geom.n_expert, 0);

    const auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < nL; ++l) {
        m.begin_layer();
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool sparse = l >= (int) m.geom.n_dense_lead;
        size_t ne0, ne1;
        const std::vector<float> *an = m.get(p + "attn_norm.weight", err), *wq = m.ne_of(p + "attn_q.weight", ne0, ne1),
                                   *wk = m.ne_of(p + "attn_k.weight", ne0, ne1), *wv = m.ne_of(p + "attn_v.weight", ne0, ne1),
                                   *qn = m.get(p + "attn_q_norm.weight", err), *kn = m.get(p + "attn_k_norm.weight", err),
                                   *wo = m.ne_of(p + "attn_output.weight", ne0, ne1);
        if (!wq || !wk || !wv || !wo || !an) { std::fprintf(stderr, "layer %d: %s\n", l, err.c_str()); return 1; }
        kv[l].reserve(HKV, HD, T + 8);
        if (sparse) ik_cache.assign((size_t) (T + 8) * m.idx.dim, 0.0f);

        const std::vector<float> *iqw = nullptr, *ikw = nullptr, *iqn = nullptr, *ikn = nullptr;
        if (sparse) {
            size_t a, b;
            iqw = m.ne_of(p + "indexer.q_proj.weight", a, b);
            ikw = m.ne_of(p + "indexer.k_proj.weight", a, b);
            iqn = m.get(p + "indexer.q_norm.weight", err);
            ikn = m.get(p + "indexer.k_norm.weight", err);
            if (!iqw || !ikw || !iqn || !ikn) { std::fprintf(stderr, "layer %d indexer: %s\n", l, err.c_str()); return 1; }
        }

        std::vector<float> h(NE), q((size_t) H * HD), kk((size_t) HKV * HD), vv((size_t) HKV * HD),
            aout((size_t) H * HD), o(NE);
        // std::vector<bool> is bit-packed and has no data(), and both masks are read as a plain bool array
        std::unique_ptr<bool[]> vis(new bool[(size_t) T + 1]);
        for (int t = 0; t < T; ++t) {
            mm::rms_norm(x[t].data(), an->data(), h.data(), NE, (float) m.geom.rms_eps);
            mm::project_norm_rope(h.data(), m.dims, wq->data(), wk->data(), wv->data(), qn ? qn->data() : nullptr,
                                  kn ? kn->data() : nullptr, (float) t, q.data(), kk.data(), vv.data());
            for (int k = 0; k < HKV; ++k) {
                std::copy(kk.begin() + k * HD, kk.begin() + (k + 1) * HD,
                          kv[l].k.begin() + ((size_t) k * (kv[l].k.size() / (HKV * HD)) + kv[l].n_pos) * HD);
                std::copy(vv.begin() + k * HD, vv.begin() + (k + 1) * HD,
                          kv[l].v.begin() + ((size_t) k * (kv[l].v.size() / (HKV * HD)) + kv[l].n_pos) * HD);
            }
            if (sparse) {
                std::vector<float> iq((size_t) m.idx.n_groups * m.idx.dim), ikl(m.idx.dim), iqp((size_t) m.idx.n_groups * m.idx.dim);
                mm::linear(iqw->data(), h.data(), iq.data(), NE, m.idx.n_groups * m.idx.dim);
                mm::linear(ikw->data(), h.data(), ikl.data(), NE, m.idx.dim);
                mm::indexer_prep(ikl.data(), ikn->data(), m.idx.dim, m.dims.n_rot, (float) t, m.dims.rope_base, m.idx.eps,
                                 ik_cache.data() + (size_t) kv[l].n_pos * m.idx.dim);
                for (int g = 0; g < m.idx.n_groups; ++g) {
                    mm::indexer_prep(iq.data() + (size_t) g * m.idx.dim, iqn->data(), m.idx.dim, m.dims.n_rot, (float) t,
                                     m.dims.rope_base, m.idx.eps, iqp.data() + (size_t) g * m.idx.dim);
                }
                const int np = kv[l].n_pos + 1;
                std::vector<mm::BlockSel> sel((size_t) m.idx.n_groups);
                std::unique_ptr<bool[]> visall(new bool[(size_t) np]);
                for (int i = 0; i < np; ++i) visall[i] = false;
                // the key cache has no holes in this pass (a prompt is read in order), so no -INFINITY mask
                mm::msa_visible(iqp.data(), ik_cache.data(), nullptr, m.idx, np, t, sel.data(), visall.get());
                // the flat union over groups is what a CPU reference masks with; causality always applies on top
                for (int i = 0; i < np; ++i) vis[i] = visall[i] && i <= t;
                vis[kv[l].n_pos] = true;                                              // the token itself is always read
            } else {
                for (int i = 0; i <= kv[l].n_pos; ++i) vis[i] = true;
            }
            mm::attn_one(q.data(), kv[l].k.data(), kv[l].v.data(), vis.get(), m.dims, kv[l].n_pos + 1, aout.data(),
                            kv[l].cap);   // the cache is strided by its capacity, not by the positions used
            mm::linear(wo->data(), aout.data(), o.data(), H * HD, NE);
            for (int i = 0; i < NE; ++i) { x[t][i] += o[i]; if (!std::isfinite(x[t][i])) ++rep.nan_hits; }
            ++kv[l].n_pos;
        }

        // the feed-forward: dense swiglu in the leading layers, routed experts plus a shared one after them
        const std::vector<float>* fn = m.get(p + "ffn_norm.weight", err);
        std::vector<float> hn(NE), f(NE);
        mm::MoEWeights mw;
        std::vector<float> sg, su, sd, gslot, uslot, dslot;
        const std::vector<float> *dg = nullptr, *du = nullptr, *dd = nullptr, *router = nullptr, *bias = nullptr;
        if (!sparse) {
            size_t a, b;
            dg = m.ne_of(p + "ffn_gate.weight", a, b);
            du = m.ne_of(p + "ffn_up.weight", a, b);
            dd = m.ne_of(p + "ffn_down.weight", a, b);
            if (!dg || !du || !dd) { std::fprintf(stderr, "layer %d ffn: %s\n", l, err.c_str()); return 1; }
        } else {
            size_t a, b;
            router = m.ne_of(p + "ffn_gate_inp.weight", a, b);
            bias = m.get(p + "exp_probs_b.bias", err);
            const std::vector<float>* shg = m.ne_of(p + "ffn_gate_shexp.weight", a, b);
            const std::vector<float>* shu = m.ne_of(p + "ffn_up_shexp.weight", a, b);
            const std::vector<float>* shd = m.ne_of(p + "ffn_down_shexp.weight", a, b);
            if (!router || !bias || !shg || !shu || !shd) { std::fprintf(stderr, "layer %d moe: %s\n", l, err.c_str()); return 1; }
            sg = *shg; su = *shu; sd = *shd;                                            // one shared expert: resident
            mw.n_expert = (int) m.geom.n_expert; mw.n_expert_used = (int) m.geom.n_expert_used;
            mw.n_ff = (int) m.geom.n_ff; mw.n_embd = NE; mw.scale = 2.0f; mw.norm = true;
            mw.router = router->data(); mw.bias = bias->data();
            mw.sh_gate = sg.data(); mw.sh_up = su.data(); mw.sh_down = sd.data();
        }
        for (int t = 0; t < T; ++t) {
            mm::rms_norm(x[t].data(), fn ? fn->data() : nullptr, hn.data(), NE, (float) m.geom.rms_eps);
            if (!sparse) {
                mm::ffn_dense(hn.data(), NE, (int) m.geom.n_ff_dense, dg->data(), du->data(), dd->data(), f.data());
            } else {
                // the four selected experts, dequantised one slot at a time
                std::vector<float> logits((size_t) mw.n_expert);
                mm::linear(mw.router, hn.data(), logits.data(), NE, mw.n_expert);
                const mm::Route r = mm::route(logits.data(), mw.bias, mw.n_expert, mw.n_expert_used, true, 2.0f);
                if (report) {
                    std::vector<float> srt(r.weights, r.weights + r.n);
                    std::sort(srt.begin(), srt.end(), std::greater<float>());
                    for (int i = 0; i + 1 < (int) srt.size(); ++i) rep.min_gap = std::min(rep.min_gap, (double) srt[i] - (double) srt[i + 1]);
                    for (int i = 0; i < r.n; ++i) rep.sum_w += r.weights[i];
                    rep.checked += 1;
                    for (int i = 0; i < r.n; ++i) if (r.experts[i] >= 0) rep.expert_hits[r.experts[i]] += 1;
                }
                std::vector<std::vector<float>> rows((size_t) r.n, std::vector<float>(NE));
                std::vector<const float*> rowp((size_t) r.n);
                std::vector<float> hh((size_t) mw.n_ff), row(NE);
                for (int s = 0; s < r.n; ++s) {
                    if (!m.slot(l, r.experts[s], "ffn_gate_exps.weight", gslot, err) ||
                        !m.slot(l, r.experts[s], "ffn_up_exps.weight", uslot, err) ||
                        !m.slot(l, r.experts[s], "ffn_down_exps.weight", dslot, err)) {
                        std::fprintf(stderr, "layer %d expert %d: %s\n", l, r.experts[s], err.c_str());
                        return 1;
                    }
                    mm::expert_row(hn.data(), NE, mw.n_ff, gslot.data(), uslot.data(), dslot.data(), hh.data(), row.data());
                    rows[s] = row;
                    rowp[s] = rows[s].data();
                }
                mm::expert_row(hn.data(), NE, mw.n_ff, mw.sh_gate, mw.sh_up, mw.sh_down, hh.data(), row.data());
                mm::combine(r, rowp.data(), row.data(), NE, f.data());
            }
            for (int i = 0; i < NE; ++i) { x[t][i] += f[i]; if (!std::isfinite(x[t][i])) ++rep.nan_hits; }
        }
        if ((l + 1) % 10 == 0 || l + 1 == nL) {
            std::printf("layer       %3d/%d done  %.1f s\n", l + 1, nL, secs(t0));
            std::fflush(stdout);
        }
    }

    // final norm and the output head, on the last token
    size_t ne0 = 0, ne1 = 0;
    std::string e2;
    const std::vector<float>* on = m.get("output_norm.weight", e2);
    const std::vector<float>* head = m.ne_of("output.weight", ne0, ne1);
    if (!head) { std::fprintf(stderr, "output head: %s\n", e2.c_str()); return 1; }
    std::vector<float> xn(NE), logits(ne1);
    mm::rms_norm(x[T - 1].data(), on ? on->data() : nullptr, xn.data(), NE, (float) m.geom.rms_eps);
    mm::linear(head->data(), xn.data(), logits.data(), NE, (int) ne1);

    std::vector<int> order(logits.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("logits        %zu over the vocabulary in %.1f s total\n", logits.size(), secs(t0));
    std::printf("next token    ");
    for (int i = 0; i < top && i < (int) order.size(); ++i) std::printf("%d(%.4f) ", order[i], logits[order[i]]);
    std::printf("\n");
    if (report) {
        int used = 0, hot = 0;
        for (int h : rep.expert_hits) { if (h > 0) ++used; if (h > 4) ++hot; }
        std::printf("router        %d expert-selections, %d/%d experts ever picked, min gap between the 3rd and "
                    "4th weight %.3e, mean weight sum %.6f, non-finite activations %d\n",
                    rep.checked, used, (int) rep.expert_hits.size(), rep.checked ? rep.min_gap : 0.0f,
                    rep.checked ? rep.sum_w / rep.checked : 0.0, rep.nan_hits);
    }
    if (rep.nan_hits > 0) { std::printf("FAIL          %d non-finite activations\n", rep.nan_hits); return 1; }
    return 0;
}
