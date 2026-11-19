// src/minimax/m3_engine.cu - the MiniMax-M3.1-Flash engine: load the GGUF, run the forward, serve.
//
// The forward math is the one validated by tools/m3ref/m3_native (which matches the reference layer-by-layer and on
// the next token). This is a CPU forward to start: it proves the engine path end-to-end (load, run, serve, wire) and
// produces correct tokens. The CUDA kernels - projections through native_mmvq, the expert GEMV through
// native_expert, the MSA attention - are the next increment; the forward shape and the weight layout stay the same.
//
// The model is a 60-layer MoE: 3 leading dense layers, 57 sparse (128 routed experts top-4 + 1 shared), GQA 64/4
// heads at head_dim 128, partial RoPE over 64 of 128 dims, no attention output gate, and per-head top-16-of-128-
// block sparse attention (MSA) with a separate indexer.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/geometry.hpp"
#include "strata/minimax/forward.hpp"
#include "strata/minimax/engine.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mm = strata::minimax;
namespace core = strata::core;
namespace Q = strata::quant;

namespace {

double secs(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

/// One tensor's elements as floats. F32 needs no dequantising (can_dequant is false for it on purpose); everything
/// else goes through the Strata quant layer.
bool to_floats(Q::Type ty, const uint8_t* src, uint64_t n, std::vector<float>& out, std::string& err) {
    out.assign((size_t) n, 0.0f);
    if (ty == Q::Type::F32) { std::memcpy(out.data(), src, (size_t) n * sizeof(float)); return true; }
    if (!Q::can_dequant(ty)) { err = "no float path for type " + std::to_string((int) ty); return false; }
    if (!Q::dequant_row(ty, src, out.data(), (int64_t) n)) { err = "dequant failed for type " + std::to_string((int) ty); return false; }
    return true;
}

/// The weights the forward needs, read from the file as float rows and kept only while the layer using them runs.
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
        if (strata::model_family(meta) != strata::ModelFamily::MiniMaxM3) { err = "this engine takes a minimax-m3 file"; return false; }
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

    void begin_layer() {
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (it->first.rfind("blk.", 0) == 0) it = cache_.erase(it);
            else ++it;
        }
    }

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

    /// One expert of one layer's stacked tensor. The stack is the slowest dimension, so an expert's bytes are
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

/// The validated forward: embed the tokens, run every layer (attention then FFN), and return the final logits over
/// the vocabulary. `n_layers` <= 0 runs all of them.
bool forward(Model& m, const std::vector<int>& tokens, int n_layers, std::vector<float>& logits, std::string& err) {
    const int T = (int) tokens.size();
    const int nL = n_layers > 0 && n_layers < (int) m.geom.n_layers ? n_layers : (int) m.geom.n_layers;
    const int NE = m.dims.n_embd, H = m.dims.n_head, HKV = m.dims.n_head_kv, HD = m.dims.head_dim;

    std::vector<std::vector<float>> x((size_t) T);
    {
        // the embedding: [vocab, n_embd] in the file (dim 0 = n_embd), so token t's row is at t * n_embd
        const std::vector<float>* e = m.get("token_embd.weight", err);
        if (!e) return false;
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || (size_t) tokens[t] * NE + NE > e->size()) { err = "token id out of range"; return false; }
            x[t].assign(e->begin() + (long) ((size_t) tokens[t] * NE), e->begin() + (long) ((size_t) tokens[t] + 1) * NE);
        }
    }
    std::vector<KvCache> kv((size_t) nL);
    std::vector<float> ik_cache;

    for (int l = 0; l < nL; ++l) {
        m.begin_layer();
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool sparse = l >= (int) m.geom.n_dense_lead;
        size_t ne0, ne1;
        const std::vector<float> *an = m.get(p + "attn_norm.weight", err), *wq = m.get(p + "attn_q.weight", err),
                                   *wk = m.get(p + "attn_k.weight", err), *wv = m.get(p + "attn_v.weight", err),
                                   *qn = m.get(p + "attn_q_norm.weight", err), *kn = m.get(p + "attn_k_norm.weight", err),
                                   *wo = m.get(p + "attn_output.weight", err);
        if (!wq || !wk || !wv || !wo || !an) { err = "layer " + std::to_string(l) + ": " + err; return false; }
        kv[l].reserve(HKV, HD, T + 8);
        if (sparse) ik_cache.assign((size_t) (T + 8) * m.idx.dim, 0.0f);

        const std::vector<float> *iqw = nullptr, *ikw = nullptr, *iqn = nullptr, *ikn = nullptr;
        if (sparse) {
            iqw = m.get(p + "indexer.q_proj.weight", err);
            ikw = m.get(p + "indexer.k_proj.weight", err);
            iqn = m.get(p + "indexer.q_norm.weight", err);
            ikn = m.get(p + "indexer.k_norm.weight", err);
            if (!iqw || !ikw || !iqn || !ikn) { err = "layer " + std::to_string(l) + " indexer: " + err; return false; }
        }

        std::vector<float> h(NE), q((size_t) H * HD), kk((size_t) HKV * HD), vv((size_t) HKV * HD),
            aout((size_t) H * HD), o(NE);
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
                for (int g = 0; g < m.idx.n_groups; ++g)
                    mm::indexer_prep(iq.data() + (size_t) g * m.idx.dim, iqn->data(), m.idx.dim, m.dims.n_rot, (float) t,
                                     m.dims.rope_base, m.idx.eps, iqp.data() + (size_t) g * m.idx.dim);
                std::vector<mm::BlockSel> sel((size_t) m.idx.n_groups);
                std::unique_ptr<bool[]> visall(new bool[(size_t) (kv[l].n_pos + 1)]);
                for (int i = 0; i <= kv[l].n_pos; ++i) visall[i] = false;
                mm::msa_visible(iqp.data(), ik_cache.data(), nullptr, m.idx, kv[l].n_pos + 1, t, sel.data(), visall.get());
                for (int i = 0; i <= kv[l].n_pos; ++i) vis[i] = visall[i] && i <= t;
                vis[kv[l].n_pos] = true;
            } else {
                for (int i = 0; i <= kv[l].n_pos; ++i) vis[i] = true;
            }
            mm::attn_one(q.data(), kv[l].k.data(), kv[l].v.data(), vis.get(), m.dims, kv[l].n_pos + 1, aout.data(), kv[l].cap);
            mm::linear(wo->data(), aout.data(), o.data(), H * HD, NE);
            for (int i = 0; i < NE; ++i) x[t][i] += o[i];
            ++kv[l].n_pos;
        }

        const std::vector<float>* fn = m.get(p + "ffn_norm.weight", err);
        std::vector<float> hn(NE), f(NE);
        if (!sparse) {
            const std::vector<float> *dg = m.get(p + "ffn_gate.weight", err),
                                     *du = m.get(p + "ffn_up.weight", err),
                                     *dd = m.get(p + "ffn_down.weight", err);
            if (!dg || !du || !dd) { err = "layer " + std::to_string(l) + " ffn: " + err; return false; }
            for (int t = 0; t < T; ++t) {
                mm::rms_norm(x[t].data(), fn ? fn->data() : nullptr, hn.data(), NE, (float) m.geom.rms_eps);
                mm::ffn_dense(hn.data(), NE, (int) m.geom.n_ff_dense, dg->data(), du->data(), dd->data(), f.data());
                for (int i = 0; i < NE; ++i) x[t][i] += f[i];
            }
        } else {
            mm::MoEWeights mw;
            mw.n_expert = (int) m.geom.n_expert; mw.n_expert_used = (int) m.geom.n_expert_used;
            mw.n_ff = (int) m.geom.n_ff; mw.n_embd = NE; mw.scale = 2.0f; mw.norm = true;
            const std::vector<float> *router = m.get(p + "ffn_gate_inp.weight", err),
                                     *bias = m.get(p + "exp_probs_b.bias", err);
            const std::vector<float> *shg = m.get(p + "ffn_gate_shexp.weight", err),
                                     *shu = m.get(p + "ffn_up_shexp.weight", err),
                                     *shd = m.get(p + "ffn_down_shexp.weight", err);
            if (!router || !bias || !shg || !shu || !shd) { err = "layer " + std::to_string(l) + " moe: " + err; return false; }
            mw.router = router->data(); mw.bias = bias->data();
            mw.sh_gate = shg->data(); mw.sh_up = shu->data(); mw.sh_down = shd->data();
            std::vector<float> gslot, uslot, dslot;
            for (int t = 0; t < T; ++t) {
                mm::rms_norm(x[t].data(), fn ? fn->data() : nullptr, hn.data(), NE, (float) m.geom.rms_eps);
                std::vector<float> logits_e((size_t) mw.n_expert);
                mm::linear(mw.router, hn.data(), logits_e.data(), NE, mw.n_expert);
                const mm::Route r = mm::route(logits_e.data(), mw.bias, mw.n_expert, mw.n_expert_used, true, 2.0f);
                std::vector<std::vector<float>> rows((size_t) r.n, std::vector<float>(NE));
                std::vector<const float*> rowp((size_t) r.n);
                std::vector<float> hh((size_t) mw.n_ff), row(NE);
                for (int s = 0; s < r.n; ++s) {
                    if (!m.slot(l, r.experts[s], "ffn_gate_exps.weight", gslot, err) ||
                        !m.slot(l, r.experts[s], "ffn_up_exps.weight", uslot, err) ||
                        !m.slot(l, r.experts[s], "ffn_down_exps.weight", dslot, err)) {
                        err = "layer " + std::to_string(l) + " expert " + std::to_string(r.experts[s]) + ": " + err;
                        return false;
                    }
                    mm::expert_row(hn.data(), NE, mw.n_ff, gslot.data(), uslot.data(), dslot.data(), hh.data(), row.data());
                    rows[s] = row;
                    rowp[s] = rows[s].data();
                }
                mm::expert_row(hn.data(), NE, mw.n_ff, mw.sh_gate, mw.sh_up, mw.sh_down, hh.data(), row.data());
                mm::combine(r, rowp.data(), row.data(), NE, f.data());
                for (int i = 0; i < NE; ++i) x[t][i] += f[i];
            }
        }
    }

    // final norm + the output head, on the last token
    const std::vector<float>* on = m.get("output_norm.weight", err);
    const std::vector<float>* head = m.get("output.weight", err);
    if (!head) { err = "output head: " + err; return false; }
    std::vector<float> xn(NE);
    mm::rms_norm(x[T - 1].data(), on ? on->data() : nullptr, xn.data(), NE, (float) m.geom.rms_eps);
    logits.assign((size_t) m.geom.n_vocab, 0.0f);
    mm::linear(head->data(), xn.data(), logits.data(), NE, (int) m.geom.n_vocab);
    return true;
}

}  // namespace

namespace strata {
namespace minimax {

int run(int argc, char** argv) {
    std::string model, tokens;
    int ctx = 8192, n_gen = 64;
    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](std::string& d) { return i + 1 < argc ? (d = argv[++i], true) : false; };
        std::string v;
        if (k == "--model") { if (!next(model)) return 2; }
        else if (k == "--tokens") { if (!next(tokens)) return 2; }
        else if (k == "--ctx") { if (!next(v)) return 2; ctx = std::atoi(v.c_str()); }
        else if (k == "--n") { if (!next(v)) return 2; n_gen = std::atoi(v.c_str()); }
        else if (k == "--serve") serve = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: strata --native --model <shard> (--serve | --tokens 1,2,3)\n"); return 2; }

    std::string err;
    Model m;
    if (!m.open(model, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::printf("m3 engine     %lld layers, hidden %d, heads %d/%d x %d, rot %d, experts %lld top-%lld, dense lead %d\n",
                (long long) m.geom.n_layers, m.dims.n_embd, m.dims.n_head, m.dims.n_head_kv, m.dims.head_dim,
                m.dims.n_rot, (long long) m.geom.n_expert, (long long) m.geom.n_expert_used, (int) m.geom.n_dense_lead);
    std::fflush(stdout);

    if (!serve) {
        if (tokens.empty()) { std::fprintf(stderr, "--tokens is required without --serve\n"); return 2; }
        std::vector<int> ids;
        size_t p = 0;
        while (p <= tokens.size()) {
            const size_t c = tokens.find(',', p);
            const std::string s = tokens.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!s.empty()) ids.push_back(std::atoi(s.c_str()));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        std::vector<float> logits;
        const auto t0 = std::chrono::steady_clock::now();
        if (!forward(m, ids, 0, logits, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::vector<int> order(logits.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        std::printf("next token    ");
        for (int i = 0; i < 5 && i < (int) order.size(); ++i) std::printf("%d(%.4f) ", order[i], logits[order[i]]);
        std::printf("\nin %.1f s\n", secs(t0));
        return 0;
    }

    // --serve: serve/server.py's engine protocol. READY announces the context; GEN <max> [k=v] <ids> runs a prompt
    // and streams the generated ids as T lines, then a DONE line; STOP ends it.
    std::printf("READY %d stop\n", ctx);
    std::fflush(stdout);
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    for (;;) {
        std::string line;
        bool have = false;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (!lines.empty()) { line = lines.front(); lines.pop_front(); have = true; }
            else if (eof) return 0;
        }
        if (!have) continue;
        if (line.rfind("GEN", 0) != 0) { std::printf("ERR expected: GEN <max> <id,id,...>\n"); std::fflush(stdout); continue; }
        // GEN <max_new> <id,id,...>
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        const int max_new = std::atoi(line.c_str() + sp + 1);
        size_t idsp = line.find(' ', sp + 1);
        if (idsp == std::string::npos) continue;
        std::vector<int> ids;
        size_t p = idsp + 1;
        while (p <= line.size()) {
            const size_t c = line.find(',', p);
            const std::string s = line.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!s.empty()) ids.push_back(std::atoi(s.c_str()));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int> produced;
        for (int g = 0; g < max_new; ++g) {
            std::vector<float> logits;
            if (!forward(m, ids, 0, logits, err)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); break; }
            int best = 0;
            for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = (int) i;
            produced.push_back(best);
            std::printf("T %d\n", best);
            std::fflush(stdout);
            ids.push_back(best);
        }
        std::printf("DONE %d %zu %.1f 0.0 %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", (long long) produced.size(), ids.size(),
                    secs(t0), produced.empty() ? "" : "stop", ids.size(), (long long) 0);
        std::fflush(stdout);
        if (stop.load()) break;
    }
}

}  // namespace minimax
}  // namespace strata
