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
#include "strata/quant/act.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"
#include "m3_cuda.cuh"

namespace K = strata::kernels;

#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <list>
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

const char* g_dump_q = nullptr;
bool mmf_debug = false;
std::vector<float> g_dump_q_data;
const char* g_dump_x = nullptr;
const char* g_dump_xa = nullptr;
const char* g_dump_dv = nullptr;
const char* g_dump_dq = nullptr;
const char* g_dump_qn = nullptr;
const char* g_dump_qr = nullptr;
std::vector<float> g_dump_qr_data;
const char* g_dump_da = nullptr;
std::vector<float> g_dump_da_data;
const char* g_dump_kr = nullptr;
std::vector<float> g_dump_kr_data;

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

    /// The quantised bytes of a tensor (for GPU upload), its ggml type, and its shape.
    const uint8_t* raw(const std::string& name, int64_t& type, std::vector<uint64_t>& shape, std::string& err) {
        size_t si = 0;
        const strata::TensorInfo* t = file_->find(name, &si);
        if (!t) { err = "missing tensor " + name; return nullptr; }
        type = (int64_t) t->type;
        shape = t->shape;
        return file_->shard(si).tensor_data(*t);
    }
    /// One expert's dequantised F32 matrices, from the persistent LRU cache (dequantised once, reused across
    /// tokens and across forwards, so a warm cache streams over PCIe instead of re-dequantising on the CPU).
    /// One expert's dequantised F32 matrices, from the persistent LRU cache (dequantised once, reused across
    /// tokens and forwards, so a warm cache streams over PCIe instead of re-dequantising on the CPU).
    /// `gate`/`up`/`down` are set to point at the cache slot (no copy); the pointers stay valid until the slot is
    /// evicted, so the caller must finish any async copy out of them before the next call that may evict.
    bool cached_expert(int layer, int expert, int n_embd, int n_ff, const std::vector<float>** gate,
                       const std::vector<float>** up, const std::vector<float>** down, std::string& err) {
        const uint64_t key = ((uint64_t) (uint32_t) layer << 32) | (uint64_t) (uint32_t) expert;
        auto hit = lru_pos_.find(key);
        if (hit != lru_pos_.end()) {
            lru_.splice(lru_.begin(), lru_, hit->second);
            const size_t s = slot_of_.at(key);
            *gate = &expert_g_[s]; *up = &expert_u_[s]; *down = &expert_d_[s];
            return true;
        }
        // a miss: get a slot (evict the least-recently-used one if the pool is full)
        size_t slot;
        if (free_slots_.empty()) {
            const uint64_t old = lru_.back();
            slot = slot_of_[old];
            lru_pos_.erase(old);
            slot_of_.erase(old);   // the stale entry would let two live keys share a slot
            lru_.pop_back();
        } else {
            slot = free_slots_.back();
            free_slots_.pop_back();
        }
        const size_t gu = (size_t) n_embd * n_ff, dd = (size_t) n_ff * n_embd;
        // each matrix keeps its OWN ggml type (in this model gate/up are IQ1_M, down is IQ2_XXS): dequantising one
        // with another's type reads the blocks wrong and yields garbage - the old shared-`ty` did exactly that.
        int64_t gty, uty, dty; uint64_t gb, ub, db;
        const uint8_t *gh = raw_slot(layer, expert, "ffn_gate_exps.weight", gty, gb, err),
                      *uh = raw_slot(layer, expert, "ffn_up_exps.weight", uty, ub, err),
                      *dh = raw_slot(layer, expert, "ffn_down_exps.weight", dty, db, err);
        if (!gh || !uh || !dh) { err = "expert " + std::to_string(expert) + ": " + err; return false; }
        if (!Q::dequant_row((Q::Type) gty, gh, expert_g_[slot].data(), (int64_t) gu) ||
            !Q::dequant_row((Q::Type) uty, uh, expert_u_[slot].data(), (int64_t) gu) ||
            !Q::dequant_row((Q::Type) dty, dh, expert_d_[slot].data(), (int64_t) dd)) { err = "expert dequant"; return false; }
        lru_.push_front(key); lru_pos_[key] = lru_.begin(); slot_of_[key] = slot;
        *gate = &expert_g_[slot]; *up = &expert_u_[slot]; *down = &expert_d_[slot];
        return true;
    }
    size_t expert_cache_size() const { return expert_slots_; }
    /// Size the expert cache: expert_slots_ of (gate, up, down) F32 buffers in CPU RAM.  Idempotent: a second call
    /// with the same size keeps the warm cache (the Model outlives every forward).
    void init_expert_cache(size_t slots, int n_embd, int n_ff) {
        if (expert_slots_ == slots && expert_g_.size() == slots) return;
        expert_g_.clear(); expert_u_.clear(); expert_d_.clear();
        lru_.clear(); lru_pos_.clear(); slot_of_.clear();
        expert_slots_ = slots;
        free_slots_.clear();
        for (size_t i = 0; i < slots; ++i) {
            expert_g_.emplace_back((size_t) n_embd * n_ff);
            expert_u_.emplace_back((size_t) n_embd * n_ff);
            expert_d_.emplace_back((size_t) n_ff * n_embd);
            free_slots_.push_back(i);
        }
    }

    /// One expert's quantised bytes from a stacked tensor (the slowest dim is the expert), for GPU upload.
    const uint8_t* raw_slot(int layer, int expert, const char* what, int64_t& type, uint64_t& bytes, std::string& err) {
        const std::string name = "blk." + std::to_string(layer) + "." + what;
        size_t si = 0;
        const strata::TensorInfo* t = file_->find(name, &si);
        if (!t || t->shape.size() < 3) { err = "missing stacked tensor " + name; return nullptr; }
        if (expert < 0 || (uint64_t) expert >= t->shape[2]) { err = "expert out of range in " + name; return nullptr; }
        const Q::Type ty = (Q::Type) t->type;
        const size_t total = Q::row_bytes((int) ty, (int64_t) t->elements());
        if (total == 0 || t->shape[2] == 0 || total % t->shape[2] != 0) { err = name + ": unequal expert strides"; return nullptr; }
        type = (int64_t) t->type;
        bytes = total / t->shape[2];
        return file_->shard(si).tensor_data(*t) + (size_t) expert * bytes;
    }

    mm::LayerDims dims;
    mm::IndexerShape idx;
    core::ModelGeometry geom;

  private:
    std::unique_ptr<strata::GgufModel> file_;
    std::unordered_map<std::string, std::vector<float>> cache_;
    // the persistent F32 expert LRU cache (warm across forwards): expert_slots_ of (gate, up, down)
    size_t expert_slots_ = 0;
    std::list<uint64_t> lru_;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_pos_;
    std::unordered_map<uint64_t, size_t> slot_of_;
    std::vector<size_t> free_slots_;
    std::vector<std::vector<float>> expert_g_, expert_u_, expert_d_;
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
            std::vector<float> qnorm_out;
            if (g_dump_qn && l == 0 && t == 0) { std::FILE* f=std::fopen(g_dump_qn,"w"); for(int i=0;i<HD;++i) std::fprintf(f,"%.6f\n",qn?(*qn)[i]:1.0f); std::fclose(f); }
            if (g_dump_dq && l == 0 && t == T - 1) { std::vector<float> qp((size_t)H*HD); mm::linear(wq->data(), h.data(), qp.data(), NE, H*HD); std::FILE* f=std::fopen(g_dump_dq,"w"); for(int i=0;i<H*HD;++i) std::fprintf(f,"%.6f\n",qp[i]); std::fclose(f); }
            mm::project_norm_rope(h.data(), m.dims, wq->data(), wk->data(), wv->data(), qn ? qn->data() : nullptr,
                                  kn ? kn->data() : nullptr, (float) t, q.data(), kk.data(), vv.data(), &qnorm_out);
            if (g_dump_dv && l == 0 && t == T - 1) { std::FILE* f=std::fopen(g_dump_dv,"w"); for(int i=0;i<HKV*HD;++i) std::fprintf(f,"%.6f\n",vv[i]); std::fclose(f); }
            if (g_dump_q && l == 0 && t == T - 1) { g_dump_q_data.assign(qnorm_out.begin(), qnorm_out.begin() + H * HD); }
            if (g_dump_qr && l == 0 && t == T - 1) { g_dump_qr_data.assign(q.begin(), q.begin() + H * HD); }
            if (g_dump_kr && l == 0 && t == T - 1) { g_dump_kr_data.assign(kk.begin(), kk.begin() + HKV * HD); }
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
            if (g_dump_da && l == 0 && t == T - 1) { g_dump_da_data.assign(aout.begin(), aout.begin() + H * HD); }
            mm::linear(wo->data(), aout.data(), o.data(), H * HD, NE);
            for (int i = 0; i < NE; ++i) x[t][i] += o[i];
            if (g_dump_xa && l == 0 && t == T - 1) { std::FILE* f=std::fopen(g_dump_xa,"w"); for(int i=0;i<NE;++i) std::fprintf(f,"%.6f\n",x[t][i]); std::fclose(f); }
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
        if (g_dump_x && l == 0) { std::vector<float> d(NE); for (int i=0;i<NE;++i) d[i]=x[T-1][i]; std::FILE* f=std::fopen(g_dump_x,"w"); for(int i=0;i<NE;++i) std::fprintf(f,"%.6f\n",d[i]); std::fclose(f); }
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


/// The GPU forward: the same recipe as forward(), with the projections on the quantised GEMM (iq_mmvq on Q8_1
/// activations) and the elementwise / attention work on the m3k kernels. Experts are streamed from the mmap (their
/// quantised bytes) to the device one at a time - the 95.9 GiB expert stack does not fit in VRAM. Must produce the
/// same token as the CPU forward.
bool forward_gpu(Model& m, const std::vector<int>& tokens, int n_layers, bool use_quant, int expert_cache, std::vector<float>& logits, std::string& err) {
    const int T = (int) tokens.size();
    const int nL = n_layers > 0 && n_layers < (int) m.geom.n_layers ? n_layers : (int) m.geom.n_layers;
    const int NE = m.dims.n_embd, H = m.dims.n_head, HKV = m.dims.n_head_kv, HD = m.dims.head_dim;
    const int NFF = (int) m.geom.n_ff;
    // the persistent F32 expert LRU (opt-in via --experts N); idempotent, so the warm cache survives across forwards
    if (expert_cache > 0) m.init_expert_cache((size_t) expert_cache, NE, NFF);
    cudaStream_t st;
    if (cudaStreamCreate(&st) != cudaSuccess) { err = "cudaStreamCreate"; return false; }
    strata::kernels::pdl_scope() = false;   // PDL needs a PDL predecessor; our custom kernels aren't one, so keep launches ordinary
    auto die = [&](const char* w) -> bool {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); return true; }
        return false;
    };

    // device scratch, sized for the largest row (the expert intermediates are the biggest: n_ff)
    std::vector<float> router_host;
    // fresh-per-use F32 expert buffers (only when the LRU cache is off); reused across experts so the pointers
    // handed to cudaMemcpyAsync stay valid for the copy
    std::vector<float> fresh_g_, fresh_u_, fresh_d_;
    float *dx = nullptr, *dh = nullptr, *dq = nullptr, *dk = nullptr, *dv = nullptr, *da = nullptr, *do_ = nullptr,
          *dgate = nullptr, *dup = nullptr, *dhh = nullptr, *df = nullptr, *dr = nullptr, *dout = nullptr, *dkv_k = nullptr, *dkv_v = nullptr,
          *diq = nullptr, *dik = nullptr, *dikc = nullptr, *dq2 = nullptr, *dk2 = nullptr;
    unsigned char* dmask = nullptr;
    auto alloc = [&](float** p, size_t n) { return cudaMalloc(p, n * sizeof(float)) == cudaSuccess; };
    if (!alloc(&dx, (size_t) T * NE) || !alloc(&dh, (size_t) NE) || !alloc(&dq, (size_t) H * HD) ||
        !alloc(&dk, (size_t) HKV * HD) || !alloc(&dv, (size_t) HKV * HD) || !alloc(&da, (size_t) H * HD) ||
        !alloc(&do_, (size_t) NE) || !alloc(&dgate, (size_t) NFF) || !alloc(&dup, (size_t) NFF) ||
        !alloc(&dhh, (size_t) NFF) || !alloc(&df, (size_t) NE) || !alloc(&dr, 128) ||
        !alloc(&dout, (size_t) m.geom.n_vocab) || !alloc(&dkv_k, (size_t) HKV * (T + 8) * HD) ||
        !alloc(&dkv_v, (size_t) HKV * (T + 8) * HD) ||
        !alloc(&diq, (size_t) m.idx.n_groups * m.idx.dim) || !alloc(&dik, (size_t) m.idx.dim) ||
        !alloc(&dikc, (size_t) (T + 8) * m.idx.dim) || !alloc(&dq2, (size_t) H * HD) || !alloc(&dk2, (size_t) HKV * HD) ||
        cudaMalloc(&dmask, (size_t) (T + 8)) != cudaSuccess) { err = "cudaMalloc"; return false; }
    // Q8_1 scratch (one row of NE and one of NFF)
    // hq8 holds the Q8_1 of the largest projection input: the q-k-v use NE, the output uses H*HD (larger)
    size_t q8ne = K::native_q8_1_bytes((std::max)(NE, H * HD), 1), q8nff = K::native_q8_1_bytes(NFF, 1);
    void *hq8 = nullptr, *hh8 = nullptr;
    if (cudaMalloc(&hq8, q8ne) != cudaSuccess || cudaMalloc(&hh8, q8nff) != cudaSuccess) { err = "cudaMalloc q8"; return false; }
    int32_t* dtok = nullptr;
    if (cudaMalloc(&dtok, (size_t) T * sizeof(int32_t)) != cudaSuccess) { err = "cudaMalloc tok"; return false; }
    cudaMemcpyAsync(dtok, tokens.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, st);

    // embedding table to the device (quantised)
    int64_t etype; std::vector<uint64_t> eshape;
    const uint8_t* ehost = m.raw("token_embd.weight", etype, eshape, err);
    if (!ehost) return die("embed");
    const size_t erow = Q::row_bytes((int) etype, NE);   // table is [vocab, n_embd]; a token's row is n_embd long
    void* detab = nullptr;
    const size_t etab_bytes = erow * (size_t) m.geom.n_vocab;
    if (cudaMalloc(&detab, etab_bytes) != cudaSuccess) { err = "cudaMalloc embed"; return false; }
    cudaMemcpyAsync(detab, ehost, etab_bytes, cudaMemcpyHostToDevice, st);
    K::iq_embed_rows((int) etype, detab, erow, dtok, T, NE, dx, st); if (die("iq_embed_rows")) return false;

    // small weights that stay F32 on the device (norms, router bias) - uploaded once
    auto up_f32 = [&](const std::string& name, size_t n, float** d) -> bool {
        int64_t ty; std::vector<uint64_t> sh;
        const uint8_t* h = m.raw(name, ty, sh, err);
        if (!h) return false;
        std::vector<float> f(n);
        if ((Q::Type) ty == Q::Type::F32) { std::memcpy(f.data(), h, n * sizeof(float)); }
        else if (!Q::dequant_row((Q::Type) ty, h, f.data(), (int64_t) n)) return false;
        if (cudaMalloc(d, n * sizeof(float)) != cudaSuccess) return false;
        cudaMemcpyAsync(*d, f.data(), n * sizeof(float), cudaMemcpyHostToDevice, st);
        return true;
    };
    // projection weight cache (quantised, on the device) - one per projection, reused across tokens
    struct WProj { void* d = nullptr; int64_t type = 0; size_t bytes = 0; };
    WProj wq, wk, wv, wo, dg, du, dd, wrouter, sg, su, sd;
    auto up_proj = [&](const std::string& name, WProj& w) -> bool {
        int64_t ty; std::vector<uint64_t> sh;
        const uint8_t* h = m.raw(name, ty, sh, err);
        if (!h) return false;
        const int64_t n = sh.size() >= 2 ? (int64_t) sh[0] * sh[1] : 0;
        // quantised path: keep the bytes and let iq_mmvq dequant on the fly; the F32 fallback (the --f32 flag, and
        // any type iq_mmvq cannot launch) dequantises to F32 and uses the plain F32 matvec
        if (use_quant && (Q::Type) ty != Q::Type::F32) {
            w.type = ty;
            w.bytes = Q::row_bytes((int) ty, n);
            if (w.bytes == 0) { err = "no row bytes for type " + std::to_string((int) ty); return false; }
            if (cudaMalloc(&w.d, w.bytes) != cudaSuccess) return false;
            cudaMemcpyAsync(w.d, h, w.bytes, cudaMemcpyHostToDevice, st);
            return true;
        }
        std::vector<float> f((size_t) n);
        if ((Q::Type) ty == Q::Type::F32) std::memcpy(f.data(), h, (size_t) n * sizeof(float));
        else if (!Q::dequant_row((Q::Type) ty, h, f.data(), (int64_t) n)) return false;
        w.bytes = (size_t) n * sizeof(float);
        w.type = (int64_t) Q::Type::F32;
        if (cudaMalloc(&w.d, w.bytes) != cudaSuccess) return false;
        cudaMemcpyAsync(w.d, f.data(), w.bytes, cudaMemcpyHostToDevice, st);
        return true;
    };
    // dispatch a projection GEMM: F32 weight -> plain F32 matvec (on the F32 activation), else the quantised
    // iq_mmvq (on the Q8_1 activation)
    auto quant81 = [&](const float* x, void* y, int n) -> bool {
        try { K::native_quantize_q8_1(x, y, n, 1, st); }
        catch (const std::exception& e) { err = std::string("quant81: ") + e.what(); return false; }
        return !die("quant81");
    };
    // a projection GEMM: quantised weight -> iq_mmvq on the Q8_1 activation; F32 weight -> the plain F32 matvec
    auto mmf = [&](WProj& w, const float* f32a, void* q81a, float* y, int n_in, int n_out) -> bool {
        if (w.type == (int64_t) Q::Type::F32) {
            m3k::k_f32_mm<<<(n_out + 127) / 128, 128, 0, st>>>((const float*) w.d, f32a, y, n_in, n_out);
            return !die("mmf");
        }
        if (!quant81(f32a, q81a, n_in)) return false;
        try { K::iq_mmvq((int) w.type, w.d, q81a, y, n_in, n_out, 1, st); }
        catch (const std::exception& e) { err = std::string("mmf: ") + e.what(); return false; }
        return !die("mmf");
    };

    for (int l = 0; l < nL; ++l) {
        (void)l;
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool sparse = l >= (int) m.geom.n_dense_lead;
        // upload this layer's projections (quantised)
        if (!up_proj(p + "attn_q.weight", wq) || !up_proj(p + "attn_k.weight", wk) ||
            !up_proj(p + "attn_v.weight", wv) || !up_proj(p + "attn_output.weight", wo)) { err = "layer " + std::to_string(l) + " proj: " + err; return false; }
        float *an = nullptr, *qn = nullptr, *kn = nullptr, *fn = nullptr;
        if (!up_f32(p + "attn_norm.weight", NE, &an) || !up_f32(p + "attn_q_norm.weight", HD, &qn) ||
            !up_f32(p + "attn_k_norm.weight", HD, &kn) || !up_f32(p + "ffn_norm.weight", NE, &fn)) {
            err = "layer " + std::to_string(l) + " norms: " + err; return false;
        }
        // the MSA indexer (sparse layers only): one query head per GQA group, one shared key per position
        WProj widxq, widxk;
        float *idxqn = nullptr, *idxkn = nullptr;
        if (sparse) {
            if (!up_proj(p + "indexer.q_proj.weight", widxq) || !up_proj(p + "indexer.k_proj.weight", widxk) ||
                !up_f32(p + "indexer.q_norm.weight", m.idx.dim, &idxqn) || !up_f32(p + "indexer.k_norm.weight", m.idx.dim, &idxkn)) {
                err = "layer " + std::to_string(l) + " indexer: " + err; return false;
            }
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) { err = "weights upload l=" + std::to_string(l) + ": " + std::string(cudaGetErrorString(cudaGetLastError())); std::fprintf(stderr, "  [upload fail] l=%d\n", l); return false; }
        if (g_dump_qn && l == 0) { std::vector<float> d(HD); cudaMemcpyAsync(d.data(), qn, (size_t)HD*sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); std::FILE* f=std::fopen(g_dump_qn,"w"); for(int i=0;i<HD;++i) std::fprintf(f,"%.6f\n",d[i]); std::fclose(f); }
        const int cap = T + 8;
        // attention, per token (decode-shaped: one query row at a time)
        for (int t = 0; t < T; ++t) {
            // x[t] is at dx + t*NE
            float* xt = dx + (size_t) t * NE;
            dim3 gr((NE + 1023) / 1024, 1);
            m3k::k_rms<<<gr, 1024, 0, st>>>(xt, an, dh, NE, (float) m.geom.rms_eps);
            if (die("rms1")) return false;
            // qkv
            if (!mmf(wq, dh, hq8, dq, NE, H * HD)) return false;
            if (g_dump_dq && l == 0 && t == T - 1) { std::vector<float> d((size_t)H*HD); cudaMemcpyAsync(d.data(), dq, (size_t)H*HD*sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); std::FILE* f=std::fopen(g_dump_dq,"w"); for(int i=0;i<H*HD;++i) std::fprintf(f,"%.6f\n",d[i]); std::fclose(f); }
            if (!mmf(wk, dh, hq8, dk, NE, HKV * HD)) return false;
            if (!mmf(wv, dh, hq8, dv, NE, HKV * HD)) return false;
            if (g_dump_dv && l == 0 && t == T - 1) { std::vector<float> d((size_t)HKV*HD); cudaMemcpyAsync(d.data(), dv, (size_t)HKV*HD*sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); std::FILE* f=std::fopen(g_dump_dv,"w"); for(int i=0;i<HKV*HD;++i) std::fprintf(f,"%.6f\n",d[i]); std::fclose(f); }
            // qk norm (in place, per head): q is H rows of HD, k is HKV rows of HD
            m3k::k_rms<<<dim3(1, H), HD, 0, st>>>(dq, qn, dq2, HD, m.dims.eps);
            m3k::k_rms<<<dim3(1, HKV), HD, 0, st>>>(dk, kn, dk2, HD, m.dims.eps);
            if (die("qknorm")) return false;
            if (g_dump_q && l == 0 && t == T - 1) { g_dump_q_data.resize((size_t) H * HD); cudaMemcpyAsync(g_dump_q_data.data(), dq2, (size_t) H * HD * sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); }
            // rope (partial, NEOX)
            m3k::k_rope<<<H + HKV, m.dims.n_rot / 2, 0, st>>>(dq2, dk2, H, HKV, HD, m.dims.n_rot, (float) t, m.dims.rope_base);
            if (die("rope")) return false;
            if (g_dump_kr && l == 0 && t == T - 1) { g_dump_kr_data.resize((size_t) HKV * HD); cudaMemcpyAsync(g_dump_kr_data.data(), dk2, (size_t) HKV * HD * sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); }
            if (g_dump_qr && l == 0 && t == T - 1) { g_dump_qr_data.resize((size_t) H * HD); cudaMemcpyAsync(g_dump_qr_data.data(), dq2, (size_t) H * HD * sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); }
            // append to the KV cache ([heads, cap, HD] layout, matching k_attn)
            m3k::k_kv_append<<<HKV, HD, 0, st>>>(dkv_k, dk2, HKV, HD, cap, t);
            m3k::k_kv_append<<<HKV, HD, 0, st>>>(dkv_v, dv, HKV, HD, cap, t);
            if (sparse) {
                // MSA: the indexer scores the key blocks and the query reads the top-k of them
                if (!mmf(widxq, dh, hq8, diq, NE, m.idx.n_groups * m.idx.dim)) return false;
                if (!mmf(widxk, dh, hq8, dik, NE, m.idx.dim)) return false;
                m3k::k_rms<<<dim3(1, m.idx.n_groups), m.idx.dim, 0, st>>>(diq, idxqn, diq, m.idx.dim, m.dims.eps);
                m3k::k_rms<<<dim3(1, 1), m.idx.dim, 0, st>>>(dik, idxkn, dik, m.idx.dim, m.dims.eps);
                m3k::k_rope<<<m.idx.n_groups, m.dims.n_rot / 2, 0, st>>>(diq, nullptr, m.idx.n_groups, 0, m.idx.dim, m.dims.n_rot, (float) t, m.dims.rope_base);
                m3k::k_rope<<<1, m.dims.n_rot / 2, 0, st>>>(dik, nullptr, 1, 0, m.idx.dim, m.dims.n_rot, (float) t, m.dims.rope_base);
                if (die("indexer rope")) return false;
                cudaMemcpyAsync(dikc + (size_t) t * m.idx.dim, dik, (size_t) m.idx.dim * sizeof(float), cudaMemcpyDeviceToDevice, st);
                // score the blocks and select the top-k (CPU: the indexer key cache is small)
                const int npos = t + 1, nblk = (npos + m.idx.blk - 1) / m.idx.blk;
                std::vector<float> ikc((size_t) npos * m.idx.dim), diqh((size_t) m.idx.n_groups * m.idx.dim);
                cudaMemcpyAsync(ikc.data(), dikc, (size_t) npos * m.idx.dim * sizeof(float), cudaMemcpyDeviceToHost, st);
                cudaMemcpyAsync(diqh.data(), diq, diqh.size() * sizeof(float), cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
                std::vector<unsigned char> mask(npos, 0);
                std::vector<float> scores((size_t) nblk);
                for (int g = 0; g < m.idx.n_groups; ++g) {
                    mm::block_scores(diqh.data() + (size_t) g * m.idx.dim, ikc.data(), nullptr, npos, m.idx.dim, m.idx.blk, t, m.idx.local, scores.data());
                    const mm::BlockSel sel = mm::select_blocks(scores.data(), nblk, m.idx.topk);
                    for (int j = 0; j < sel.n; ++j)
                        for (int pp = sel.blocks[j] * m.idx.blk; pp < (sel.blocks[j] + 1) * m.idx.blk && pp < npos; ++pp) mask[pp] = 1;
                }
                cudaMemcpyAsync(dmask, mask.data(), (size_t) npos, cudaMemcpyHostToDevice, st);
                m3k::k_msa_attn<<<H, HD, 0, st>>>(dq2, dkv_k, dkv_v, dmask, da, H, HKV, HD, npos, cap);
            } else {
                m3k::k_attn<<<H, HD, 0, st>>>(dq2, dkv_k, dkv_v, da, H, HKV, HD, t + 1, cap, t);
            }
            if (die("attn")) return false;
            if (g_dump_da && l == 0 && t == T - 1) { g_dump_da_data.resize((size_t) H * HD); cudaMemcpyAsync(g_dump_da_data.data(), da, (size_t) H * HD * sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); }
            // output projection
            if (!mmf(wo, da, hq8, do_, H * HD, NE)) return false;
            m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, do_, NE);
            if (die("add1")) return false;
            if (g_dump_xa && l == 0 && t == T - 1) { std::vector<float> d(NE); cudaMemcpyAsync(d.data(), xt, (size_t)NE*sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); std::FILE* f=std::fopen(g_dump_xa,"w"); for(int i=0;i<NE;++i) std::fprintf(f,"%.6f\n",d[i]); std::fclose(f); }
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) { err = "attn loop end: " + std::string(cudaGetErrorString(cudaGetLastError())); std::fprintf(stderr, "  [l=%d] attn loop fault\n", l); return false; }
        // FFN
        if (!sparse) {
            if (!up_proj(p + "ffn_gate.weight", dg) || !up_proj(p + "ffn_up.weight", du) ||
                !up_proj(p + "ffn_down.weight", dd)) { err = "layer " + std::to_string(l) + " ffn: " + err; return false; }
            for (int t = 0; t < T; ++t) {
                float* xt = dx + (size_t) t * NE;
                dim3 gr((NE + 1023) / 1024, 1);
                m3k::k_rms<<<gr, 1024, 0, st>>>(xt, fn, dh, NE, (float) m.geom.rms_eps);
                    if (!mmf(dg, dh, hq8, dgate, NE, NFF)) return false;
                if (!mmf(du, dh, hq8, dup, NE, NFF)) return false;
                m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFF);
                if (!mmf(dd, dhh, hh8, df, NFF, NE)) return false;
                m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, df, NE);
                if (die("dense_ffn")) return false;
            }
        } else {
            // the router is tiny (n_embd x n_expert) and n_expert=128 is below iq_mmvq's launch floor, so compute it
            // on the host (dequant to F32, one copy of the activation per token)
            { int64_t rty; std::vector<uint64_t> rsh; const uint8_t* rh = m.raw(p + "ffn_gate_inp.weight", rty, rsh, err);
              if (!rh) { err = "router"; return false; }
              router_host.assign((size_t) NE * m.geom.n_expert, 0.0f);
              if ((Q::Type) rty == Q::Type::F32) std::memcpy(router_host.data(), rh, router_host.size() * sizeof(float));
              else if (!Q::dequant_row((Q::Type) rty, rh, router_host.data(), (int64_t) router_host.size())) { err = "router dequant"; return false; } }
            if (!up_proj(p + "ffn_gate_shexp.weight", sg) || !up_proj(p + "ffn_up_shexp.weight", su) ||
                !up_proj(p + "ffn_down_shexp.weight", sd)) { err = "layer " + std::to_string(l) + " shared: " + err; return false; }
            if (cudaStreamSynchronize(st) != cudaSuccess) { err = "shared upload sync: " + std::string(cudaGetErrorString(cudaGetLastError())); std::fprintf(stderr, "  [l=%d sg.type=%lld sg.bytes=%llu]\n", l, (long long) sg.type, (unsigned long long) sg.bytes); return false; }
            // the bias is applied by route() on the host, so it must be host memory (a device pointer here would segfault)
            int64_t bty; std::vector<uint64_t> bsh;
            const uint8_t* bhost = m.raw(p + "exp_probs_b.bias", bty, bsh, err);
            std::vector<float> bias_host((size_t) m.geom.n_expert, 0.0f);
            if (bhost) {
                if ((Q::Type) bty == Q::Type::F32) std::memcpy(bias_host.data(), bhost, (size_t) m.geom.n_expert * sizeof(float));
                else if (!Q::dequant_row((Q::Type) bty, bhost, bias_host.data(), (int64_t) m.geom.n_expert)) { err = "bias dequant"; return false; }
            }
            if (cudaStreamSynchronize(st) != cudaSuccess) { err = "bias sync: " + std::string(cudaGetErrorString(cudaGetLastError())); return false; }
            { cudaError_t e = cudaDeviceSynchronize(); size_t fm = 0, tm = 0; cudaMemGetInfo(&fm, &tm);
                if (e != cudaSuccess) { err = "devsync: " + std::string(cudaGetErrorString(e)); return false; } }
            // fixed expert-buffer pool (reused for every expert; the stream orders the GEMMs, so no per-expert
            // malloc/free - freeing before the async GEMM finishes would make it read freed memory)
            // all-F32 path: the expert pool holds one expert's dequantised F32 gate/up/down (streamed one at a time)
            void *exp_g = nullptr, *exp_u = nullptr, *exp_d = nullptr;
            {
                const size_t gu = (size_t) NE * NFF * sizeof(float), dd = (size_t) NFF * NE * sizeof(float);
                if (cudaMalloc(&exp_g, gu) != cudaSuccess || cudaMalloc(&exp_u, gu) != cudaSuccess ||
                    cudaMalloc(&exp_d, dd) != cudaSuccess) { err = "cudaMalloc expert pool"; return false; }
            }
            for (int t = 0; t < T; ++t) {
                float* xt = dx + (size_t) t * NE;
                dim3 gr((NE + 1023) / 1024, 1);
                m3k::k_rms<<<gr, 1024, 0, st>>>(xt, fn, dh, NE, (float) m.geom.rms_eps);
                std::vector<float> dh_host((size_t) NE);
                cudaMemcpyAsync(dh_host.data(), dh, (size_t) NE * sizeof(float), cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
                std::vector<float> rl((size_t) m.geom.n_expert, 0.0f);
                for (int e = 0; e < (int) m.geom.n_expert; ++e) { double acc = 0.0; const float* row = router_host.data() + (size_t) e * NE; for (int i = 0; i < NE; ++i) acc += (double) row[i] * dh_host[i]; rl[e] = (float) acc; }
                mm::MoEWeights mw;
                mw.n_expert = (int) m.geom.n_expert; mw.n_expert_used = (int) m.geom.n_expert_used;
                mw.n_ff = NFF; mw.n_embd = NE; mw.scale = 2.0f; mw.norm = true;
                const mm::Route r = mm::route(rl.data(), bias_host.data(), mw.n_expert, mw.n_expert_used, true, 2.0f);
                // f = 0; accumulate the routed experts (streamed) and the shared expert
                cudaMemsetAsync(df, 0, (size_t) NE * sizeof(float), st);
                for (int s = 0; s < r.n; ++s) {
                    // the expert's three matrices to F32, streamed to the pool.  With --experts N this is the
                    // persistent LRU (dequantised once, then the cached slot is re-streamed over PCIe); without it,
                    // a fresh dequant per use.  Sync first so a prior async copy out of a slot is done before this
                    // call may evict and overwrite it.
                    if (cudaStreamSynchronize(st) != cudaSuccess) { err = "expert cache sync"; return false; }
                    const std::vector<float> *gslot, *uslot, *dslot;
                    if (expert_cache > 0) {
                        if (!m.cached_expert(l, r.experts[s], NE, NFF, &gslot, &uslot, &dslot, err)) { err = "expert " + std::to_string(r.experts[s]) + ": " + err; return false; }
                    } else {
                        int64_t gty, uty, dty; uint64_t gb, ub, db;
                        const uint8_t *gh = m.raw_slot(l, r.experts[s], "ffn_gate_exps.weight", gty, gb, err),
                                      *uh = m.raw_slot(l, r.experts[s], "ffn_up_exps.weight", uty, ub, err),
                                      *dhb = m.raw_slot(l, r.experts[s], "ffn_down_exps.weight", dty, db, err);
                        if (!gh || !uh || !dhb) { err = "expert " + std::to_string(r.experts[s]) + ": " + err; return false; }
                        const uint64_t per = (uint64_t) NE * NFF;
                        fresh_g_.resize(per); fresh_u_.resize(per); fresh_d_.resize((uint64_t) NFF * NE);   // dequant overwrites all
                        if (!Q::dequant_row((Q::Type) gty, gh, fresh_g_.data(), (int64_t) per) ||
                            !Q::dequant_row((Q::Type) uty, uh, fresh_u_.data(), (int64_t) per) ||
                            !Q::dequant_row((Q::Type) dty, dhb, fresh_d_.data(), (int64_t) (NFF * NE))) { err = "expert dequant"; return false; }
                        gslot = &fresh_g_; uslot = &fresh_u_; dslot = &fresh_d_;
                    }
                    cudaMemcpyAsync(exp_g, gslot->data(), gslot->size() * sizeof(float), cudaMemcpyHostToDevice, st);
                    cudaMemcpyAsync(exp_u, uslot->data(), uslot->size() * sizeof(float), cudaMemcpyHostToDevice, st);
                    cudaMemcpyAsync(exp_d, dslot->data(), dslot->size() * sizeof(float), cudaMemcpyHostToDevice, st);
                    m3k::k_f32_mm<<<(NFF + 127) / 128, 128, 0, st>>>((const float*) exp_g, dh, dgate, NE, NFF); if (die("expert gate")) return false;
                    m3k::k_f32_mm<<<(NFF + 127) / 128, 128, 0, st>>>((const float*) exp_u, dh, dup, NE, NFF); if (die("expert up")) return false;
                    m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFF);
                    m3k::k_f32_mm<<<(NE + 127) / 128, 128, 0, st>>>((const float*) exp_d, dhh, do_, NFF, NE); if (die("expert down")) return false;
                    // f += w_s * row
                    float w = r.weights[s];
                    m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, do_, w, NE);
                    if (die("expert")) return false;
                }
                // shared expert (weight 1)
                    if (!mmf(sg, dh, hq8, dgate, NE, NFF)) return false;
                if (!mmf(su, dh, hq8, dup, NE, NFF)) return false;
                m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFF);
                if (!mmf(sd, dhh, hh8, do_, NFF, NE)) return false;
                m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, do_, 1.0f, NE);
                // the weights are already normalised by route(); f is the weighted sum + shared
                m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, df, NE);
                if (die("moe_ffn")) return false;
            }
            cudaFreeAsync(exp_g, st); cudaFreeAsync(exp_u, st); cudaFreeAsync(exp_d, st);
        }
        // NaN guard: find the first layer that produces a non-finite activation
        { std::vector<float> chk(NE); cudaMemcpyAsync(chk.data(), dx + (size_t)(T - 1) * NE, (size_t) NE * sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st);
          if (g_dump_x && l == 0) { std::FILE* f=std::fopen(g_dump_x,"w"); for(int i=0;i<NE;++i) std::fprintf(f,"%.6f\n",chk[i]); std::fclose(f); }
          bool bad = false; for (int i = 0; i < NE; ++i) if (!std::isfinite(chk[i])) { bad = true; break; }
          if (bad) { err = "non-finite activation at layer " + std::to_string(l); return false; } }
        // free this layer's projection weights (the next layer has its own)
        for (WProj* w : {&wq, &wk, &wv, &wo, &dg, &du, &dd, &wrouter, &sg, &su, &sd}) if (w->d) { cudaFreeAsync(w->d, st); w->d = nullptr; }
        for (float* f : {an, qn, kn, fn}) if (f) cudaFreeAsync(f, st);
    }

    // final norm + head, on the last token
    float* d_on = nullptr;
    if (!up_f32("output_norm.weight", NE, &d_on)) { err = "output_norm: " + err; return false; }
    WProj head;
    if (!up_proj("output.weight", head)) { err = "output head: " + err; return false; }
    float* xt = dx + (size_t) (T - 1) * NE;
    m3k::k_rms<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, d_on, dh, NE, (float) m.geom.rms_eps);
    if (!mmf(head, dh, hq8, dout, NE, (int) m.geom.n_vocab)) { err = "head: " + err; return false; }
    logits.assign((size_t) m.geom.n_vocab, 0.0f);
    cudaMemcpyAsync(logits.data(), dout, (size_t) m.geom.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (die("head")) return false;
    // free
    cudaFree(detab); cudaFree(dtok); cudaFree(dout); cudaFree(dkv_k); cudaFree(dkv_v);
    cudaFree(dx); cudaFree(dh); cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(da); cudaFree(do_);
    cudaFree(dgate); cudaFree(dup); cudaFree(dhh); cudaFree(df); cudaFree(dr);
    cudaFree(hq8); cudaFree(hh8);
    if (d_on) cudaFree(d_on);
    if (head.d) cudaFree(head.d);
    cudaStreamDestroy(st);
    return true;
}

}  // namespace

namespace strata {
namespace minimax {

int run(int argc, char** argv) {
    std::string model, tokens;
    int ctx = 8192, n_gen = 64;
    bool serve = false, use_gpu = true, use_quant = false; int n_layers = 0; int expert_cache = 0; std::string dump_logits, dump_q, dump_x, dump_xa, dump_dv, dump_dq, dump_qn, dump_qr, dump_da, dump_kr;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](std::string& d) { return i + 1 < argc ? (d = argv[++i], true) : false; };
        std::string v;
        if (k == "--model") { if (!next(model)) return 2; }
        else if (k == "--tokens") { if (!next(tokens)) return 2; }
        else if (k == "--ctx") { if (!next(v)) return 2; ctx = std::atoi(v.c_str()); }
        else if (k == "--n") { if (!next(v)) return 2; n_gen = std::atoi(v.c_str()); }
        else if (k == "--serve") serve = true;
        else if (k == "--cpu") use_gpu = false;
        else if (k == "--quant") use_quant = true;   // the quantised projection path (faster in principle; see the plan)
        else if (k == "--experts") { if (!next(v)) return 2; expert_cache = std::atoi(v.c_str()); }
        else if (k == "--layers") { if (!next(v)) return 2; n_layers = std::atoi(v.c_str()); }
        else if (k == "--dump-logits") { if (!next(dump_logits)) return 2; }
        else if (k == "--dump-q") { if (!next(dump_q)) return 2; }
        else if (k == "--dump-x") { if (!next(dump_x)) return 2; }
        else if (k == "--dump-xa") { if (!next(dump_xa)) return 2; }
        else if (k == "--dump-dv") { if (!next(dump_dv)) return 2; }
        else if (k == "--dump-dq") { if (!next(dump_dq)) return 2; }
        else if (k == "--dump-qn") { if (!next(dump_qn)) return 2; }
        else if (k == "--dump-qr") { if (!next(dump_qr)) return 2; }
        else if (k == "--dump-da") { if (!next(dump_da)) return 2; }
        else if (k == "--dump-kr") { if (!next(dump_kr)) return 2; }
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
        g_dump_q = dump_q.empty() ? nullptr : dump_q.c_str();
        g_dump_x = dump_x.empty() ? nullptr : dump_x.c_str();
        g_dump_xa = dump_xa.empty() ? nullptr : dump_xa.c_str();
        g_dump_dv = dump_dv.empty() ? nullptr : dump_dv.c_str();
        g_dump_dq = dump_dq.empty() ? nullptr : dump_dq.c_str();
        g_dump_qn = dump_qn.empty() ? nullptr : dump_qn.c_str();
        g_dump_qr = dump_qr.empty() ? nullptr : dump_qr.c_str();
        g_dump_da = dump_da.empty() ? nullptr : dump_da.c_str();
        g_dump_kr = dump_kr.empty() ? nullptr : dump_kr.c_str();
        if (!(use_gpu ? forward_gpu(m, ids, n_layers, use_quant, expert_cache, logits, err) : forward(m, ids, n_layers, logits, err))) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (!dump_logits.empty()) { std::FILE* dl = std::fopen(dump_logits.c_str(), "w"); if (dl) { for (size_t i = 0; i < logits.size(); ++i) std::fprintf(dl, "%.6f\n", logits[i]); std::fclose(dl); } }
        if (!dump_q.empty() && !g_dump_q_data.empty()) { std::FILE* dqf = std::fopen(dump_q.c_str(), "w"); if (dqf) { for (size_t i = 0; i < g_dump_q_data.size(); ++i) std::fprintf(dqf, "%.6f\n", g_dump_q_data[i]); std::fclose(dqf); } }
        if (!dump_qr.empty() && !g_dump_qr_data.empty()) { std::FILE* drf = std::fopen(dump_qr.c_str(), "w"); if (drf) { for (size_t i = 0; i < g_dump_qr_data.size(); ++i) std::fprintf(drf, "%.6f\n", g_dump_qr_data[i]); std::fclose(drf); } }
        if (!dump_da.empty() && !g_dump_da_data.empty()) { std::FILE* daf = std::fopen(dump_da.c_str(), "w"); if (daf) { for (size_t i = 0; i < g_dump_da_data.size(); ++i) std::fprintf(daf, "%.6f\n", g_dump_da_data[i]); std::fclose(daf); } }
        if (!dump_kr.empty() && !g_dump_kr_data.empty()) { std::FILE* krf = std::fopen(dump_kr.c_str(), "w"); if (krf) { for (size_t i = 0; i < g_dump_kr_data.size(); ++i) std::fprintf(krf, "%.6f\n", g_dump_kr_data[i]); std::fclose(krf); } }
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
            if (!(use_gpu ? forward_gpu(m, ids, 0, use_quant, expert_cache, logits, err) : forward(m, ids, 0, logits, err))) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); break; }
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
