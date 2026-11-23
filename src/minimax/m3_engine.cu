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
#include <algorithm>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
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

/// Next-token choice: greedy when temperature <= 0 (the engine's default), else temperature-scaled
/// min_p -> top-k -> top-p sampling, the order the engines of this family use.  The serve protocol sends the
/// keys on the GEN line (serve/server.py's sampling_keys), so a request that asks for sampling gets sampling
/// and a request that does not stays greedy.
int pick_token(const std::vector<float>& logits, double temperature, float top_p, int top_k, float min_p,
               std::mt19937& rng) {
    const size_t V = logits.size();
    int best = 0;
    for (size_t i = 1; i < V; ++i) if (logits[i] > logits[best]) best = (int) i;
    if (!(temperature > 0.0)) return best;
    std::vector<double> w(V);
    const double mx = logits[best];
    for (size_t i = 0; i < V; ++i) w[i] = std::exp((double) logits[i] - mx) / temperature;
    if (min_p > 0.0f) {
        const double lo = min_p * w[best];
        for (size_t i = 0; i < V; ++i) if (w[i] < lo) w[i] = 0.0;
    }
    std::vector<size_t> idx(V);
    for (size_t i = 0; i < V; ++i) idx[i] = i;
    if (top_k > 0 && (size_t) top_k < V) {
        std::partial_sort(idx.begin(), idx.begin() + (size_t) top_k, idx.end(),
                          [&](size_t a, size_t b) { return w[a] > w[b]; });
        for (size_t k = (size_t) top_k; k < V; ++k) w[idx[k]] = 0.0;
    }
    if (top_p < 1.0f) {
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return w[a] > w[b]; });
        double total = 0.0;
        for (size_t i = 0; i < V; ++i) total += w[idx[i]];
        double keep = 0.0;
        size_t kept = 0;
        for (size_t i = 0; i < V; ++i) {
            keep += w[idx[i]];
            ++kept;
            if (keep >= top_p * total) break;
        }
        for (size_t i = kept; i < V; ++i) w[idx[i]] = 0.0;
    }
    std::discrete_distribution<int> dd(w.begin(), w.end());
    return dd(rng);
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

// The model's shards in pinned host RAM (see Model::stage_pinned).  A file-scope instance so the staging
// survives across forwards / decode steps.
struct PinnedModel {
    std::vector<void*> pinned;   // one pinned allocation per shard
    std::vector<size_t> sizes;
    bool ready = false;
    std::mutex mu;               // stage() may be called from the serve warm-up and from a decode step
    void* shard_ptr(size_t i) const { return (ready && i < pinned.size()) ? pinned[i] : nullptr; }
    bool stage(const strata::GgufModel& g, std::string& err) {
        std::lock_guard<std::mutex> lk(mu);
        if (ready) return true;
        const size_t n = g.size();
        pinned.resize(n, nullptr);
        sizes.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const size_t sz = (size_t) g.shard(i).file_size();
            sizes[i] = sz;
            // Portable: the extras (GPU1/GPU2) DMA-read this RAM for their expert caches.  Default flags
            // register it only with the allocating device (GPU0), so the other GPUs' DMA engines read garbage.
            if (cudaHostAlloc(&pinned[i], sz, cudaHostAllocPortable) != cudaSuccess) {
                std::fprintf(stderr, "pinned stage  shard %zu (%zu MB): cudaHostAlloc failed, staying on the mmap\n", i, sz >> 20);
                for (size_t j = 0; j < i; ++j) { cudaFreeHost(pinned[j]); pinned[j] = nullptr; }
                pinned.clear(); sizes.clear();
                err = "cudaHostAlloc";
                return false;
            }
            std::fprintf(stderr, "pinned stage  shard %zu: %zu MB (one-time copy at CPU speed)\n", i, sz >> 20);
            std::fflush(stderr);
            std::memcpy(pinned[i], g.shard(i).base(), sz);
        }
        ready = true;
        return true;
    }
};
static PinnedModel g_pinned_model;

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

    /// raw_slot() pointed at pinned RAM when the model is staged there (faster H2D), else at the mmap.
    const uint8_t* raw_slot_fast(int layer, int expert, const char* what, int64_t& type, uint64_t& bytes, std::string& err) const {
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
        const strata::GgufFile& sh = file_->shard(si);
        const uint8_t* mmap_ptr = sh.tensor_data(*t) + (size_t) expert * bytes;
        if (void* pp = g_pinned_model.shard_ptr(si)) {
            const uint64_t off = (uint64_t)(mmap_ptr - sh.base());
            return (const uint8_t*) pp + off;
        }
        return mmap_ptr;
    }

    /// Stage the whole model in pinned (page-locked) host RAM so expert H2D copies run at the pinned rate
    /// (~6 GB/s on the x8 link) instead of the pageable-mmap rate (~2.9 GB/s).  One-time cost: the file size
    /// copied at CPU speed.  Falls back to the mmap when the pinned allocation fails.
    bool stage_pinned(std::string& err) { return g_pinned_model.stage(*file_, err); }

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


// ---------------------------------------------------------------------------
// VRAM expert cache (quant path): the quantised bytes of hot experts live in a persistent GPU arena, so a token
// whose routing hits a cached expert does no PCIe transfer at all.  A miss streams the expert's quantised bytes
// from the model file (page cache) through a pinned staging buffer.  Copies and GEMVs share the caller's stream,
// so stream ordering guarantees a recycled slot is never overwritten while a GEMV still reading it is in flight.
// The arena outlives every forward (a file-scope instance), so the cache is warm across generated tokens.
//
// Eviction is frequency-aware because a single token's working set (228 unique (layer,expert) instances at
// top-4 x 57 layers) exceeds any practical cache, so plain LRU thrashes: it evicts an expert the token will
// still need at a later layer.  The fix is to make RECURRING experts (the ones reused across tokens - the only
// ones a cache can ever save) unevictable:
//   - every use increments the key's frequency;
//   - a key used >= kPinThreshold times becomes pinned (never evicted); the pin set is capped at half the
//     arena so the cold tier can still hold a token's 228-instance working set;
//   - refresh() (called after a prompt pass, whose routing is free data about generation's needs) re-points the
//     cache at the highest-frequency experts, evicting only strictly colder unpinned ones - i.e. it accepts
//     that not all weights are cached and keeps the hottest fraction resident.
// ---------------------------------------------------------------------------
struct VramExpertCache {
    void* arena = nullptr;      // n_slots * slot_bytes of quantised expert bytes, [gate | up | down] per slot
    void* staging = nullptr;    // pinned host staging of slot_bytes
    size_t slot_bytes = 0;
    size_t n_slots = 0;
    struct Entry { int gty = 0, uty = 0, dty = 0; size_t gb = 0, ub = 0, db = 0; };
    std::vector<Entry> info;
    std::vector<size_t> free_slots;
    std::list<uint64_t> lru;   // key = (layer<<32)|expert; MRU at the front
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_pos;
    std::unordered_map<uint64_t, size_t> slot_of;
    std::unordered_map<uint64_t, uint32_t> freq;   // uses per key (hits + misses)
    std::unordered_set<uint64_t> pinned;           // hot experts, never evicted
    uint64_t hits = 0, misses = 0;
    int dev = 0;                                   // the GPU the arena lives on (multi-GPU: one cache per extra GPU)
    static constexpr uint32_t kPinThreshold = 8;   // uses that promote a key to pinned

    static uint64_t key(int layer, int expert) { return (uint64_t)(uint32_t) layer << 32 | (uint32_t) expert; }
    static void split(uint64_t k, int& layer, int& expert) {
        layer = (int)(k >> 32); expert = (int)(k & 0xffffffffu);
    }

    bool init(size_t slots, size_t sbytes, int device = 0) {
        if (arena) return true;   // already warm
        n_slots = slots; slot_bytes = sbytes; dev = device;
        if (slots == 0) return true;
        cudaSetDevice(dev);
        const cudaError_t e0 = cudaGetLastError();   // flush any sticky error from the device switch
        (void) e0;
        if (cudaMalloc(&arena, slots * sbytes) != cudaSuccess) {
            std::fprintf(stderr, "  [vexp init] cudaMalloc %zu MB on dev %d: %s\n",
                         (slots * sbytes) >> 20, dev, cudaGetErrorString(cudaGetLastError()));
            arena = nullptr; return false;
        }
        if (cudaHostAlloc(&staging, sbytes, cudaHostAllocDefault) != cudaSuccess) {
            std::fprintf(stderr, "  [vexp init] cudaHostAlloc %zu MB: %s\n",
                         sbytes >> 20, cudaGetErrorString(cudaGetLastError()));
            cudaFree(arena); arena = nullptr; return false;
        }
        info.resize(slots);
        for (size_t i = 0; i < slots; ++i) free_slots.push_back(i);
        return true;
    }

    // Stream expert (layer, expert)'s quantised bytes into `slot` (caller has secured it).  When the model is
    // staged in pinned RAM the source is already page-locked, so the bytes go straight to the device.
    bool load(Model& m, int layer, int expert, size_t slot, cudaStream_t st, std::string& err) {
        int64_t gt, ut, dt; uint64_t gb, ub, db;
        const uint8_t *gh = m.raw_slot_fast(layer, expert, "ffn_gate_exps.weight", gt, gb, err),
                      *uh = m.raw_slot_fast(layer, expert, "ffn_up_exps.weight", ut, ub, err),
                      *dh = m.raw_slot_fast(layer, expert, "ffn_down_exps.weight", dt, db, err);
        if (!gh || !uh || !dh) return false;
        if (gb + ub + db > slot_bytes) { err = "expert does not fit in the vexp slot"; return false; }
        uint8_t* dst = (uint8_t*) arena + slot * slot_bytes;
        if (g_pinned_model.ready) {
            if (cudaMemcpyAsync(dst, gh, (size_t) gb, cudaMemcpyHostToDevice, st) != cudaSuccess ||
                cudaMemcpyAsync(dst + gb, uh, (size_t) ub, cudaMemcpyHostToDevice, st) != cudaSuccess ||
                cudaMemcpyAsync(dst + gb + ub, dh, (size_t) db, cudaMemcpyHostToDevice, st) != cudaSuccess) {
                err = "vexp H2D"; return false;
            }
        } else {
            uint8_t* sp = (uint8_t*) staging;
            std::memcpy(sp, gh, (size_t) gb);
            std::memcpy(sp + gb, uh, (size_t) ub);
            std::memcpy(sp + gb + ub, dh, (size_t) db);
            if (cudaMemcpyAsync(dst, staging, gb + ub + db, cudaMemcpyHostToDevice, st) != cudaSuccess) { err = "vexp H2D"; return false; }
        }
        info[slot] = { (int) gt, (int) ut, (int) dt, gb, ub, db };
        return true;
    }

    // Secure a slot for a new key: a free one, or evict the least-recently-used UNPINNED key.  The pin cap
    // (n_slots/2) guarantees an unpinned victim always exists.
    bool take_slot(size_t& slot, std::string& err) {
        if (!free_slots.empty()) { slot = free_slots.back(); free_slots.pop_back(); return true; }
        auto it = lru.end();
        for (;;) { --it; if (!pinned.count(*it)) break; }   // terminates: >= half the keys are unpinned
        const uint64_t old = *it;
        lru.erase(it); lru_pos.erase(old);
        slot = slot_of[old]; slot_of.erase(old);   // the stale entry would let two keys share a slot
        return true;
    }

    // The device pointers + ggml types of expert (layer, expert)'s quantised gate/up/down bytes.  A miss evicts
    // an unpinned LRU victim (or takes a free slot) and streams the bytes in.
    bool get(Model& m, int layer, int expert, cudaStream_t st,
             const uint8_t** g, int* gty, const uint8_t** u, int* uty, const uint8_t** d, int* dty, std::string& err) {
        const uint64_t k = key(layer, expert);
        ++freq[k];
        if (!pinned.count(k) && freq[k] >= kPinThreshold && pinned.size() < n_slots / 2) pinned.insert(k);
        auto hit = lru_pos.find(k);
        size_t slot;
        if (hit != lru_pos.end()) {
            lru.splice(lru.begin(), lru, hit->second);
            slot = slot_of.at(k);
            ++hits;
        } else {
            // the CPU memcpy in load() overwrites staging; make sure the previous miss's H2D has consumed it
            if (cudaStreamSynchronize(st) != cudaSuccess) { err = "vexp staging sync"; return false; }
            if (!take_slot(slot, err)) return false;
            if (!load(m, layer, expert, slot, st, err)) return false;
            lru.push_front(k); lru_pos[k] = lru.begin(); slot_of[k] = slot;
            ++misses;
        }
        const Entry& e = info[slot];
        const uint8_t* base = (const uint8_t*) arena + slot * slot_bytes;
        *g = base;             *gty = e.gty;
        *u = base + e.gb;      *uty = e.uty;
        *d = base + e.gb + e.ub; *dty = e.dty;
        return true;
    }

    // LFU refresh: re-point the cache at the highest-frequency experts.  For each uncached key in descending
    // frequency order, load it into a free slot or by evicting the coldest UNPINNED cached key - but only if
    // that key is strictly colder (once nothing colder remains, the cache already holds the top-f experts and
    // the rest of the ranking cannot improve it).  Used after a prompt pass, whose routing is free data about
    // which experts generation will want.  Returns the number of experts loaded.
    size_t refresh(Model& m, cudaStream_t st, std::string& err) {
        std::vector<std::pair<uint32_t, uint64_t>> ranked;   // (freq, key)
        ranked.reserve(freq.size());
        for (const auto& kv : freq) ranked.emplace_back(kv.second, kv.first);
        std::sort(ranked.begin(), ranked.end(), std::greater<std::pair<uint32_t, uint64_t>>());
        size_t loaded = 0;
        for (const auto& rf : ranked) {
            const uint32_t f = rf.first;
            const uint64_t k = rf.second;
            if (lru_pos.count(k)) continue;   // already resident
            size_t slot = 0;
            if (!free_slots.empty()) {
                slot = free_slots.back(); free_slots.pop_back();
            } else {
                // the coldest unpinned cached key; skip the refresh if it is not strictly colder than `k`
                uint64_t victim = 0; uint32_t worst = 0; bool have = false;
                for (const uint64_t ck : lru) {
                    if (pinned.count(ck)) continue;
                    const uint32_t cf = freq.count(ck) ? freq[ck] : 0;
                    if (!have || cf < worst) { worst = cf; victim = ck; have = true; }
                }
                if (!have || worst >= f) break;   // the cache holds the top-frequency experts
                lru.erase(lru_pos[victim]); lru_pos.erase(victim);
                slot = slot_of[victim]; slot_of.erase(victim);
            }
            int layer, expert;
            split(k, layer, expert);
            if (cudaStreamSynchronize(st) != cudaSuccess) { err = "vexp refresh sync"; return loaded; }
            if (!load(m, layer, expert, slot, st, err)) return loaded;
            lru.push_front(k); lru_pos[k] = lru.begin(); slot_of[k] = slot;
            ++loaded;
        }
        cudaStreamSynchronize(st);
        return loaded;
    }
};
static VramExpertCache g_vexp;

// ---- Long-context KV: the sparse layers' K/V and indexer keys live in PINNED RAM (F16 K/V, F32 keys);
// each token's MSA selection (top-k blocks per GQA group, union <= kPoolBlocks) is gathered into a per-layer
// GPU block pool, so attention reads only a few MB per layer instead of the whole prefix.  The dense-lead
// layers keep their full KV on the GPU (few layers, full attention, small total).
struct RamKv {
    static constexpr int kPoolBlocks = 64;   // indexer.top_k (16) x n_groups (4): the union always fits
    // F32, not F16: this MoE's routing flips on F16 KV rounding and the output collapses to token salad.
    // The per-layer block pools are SHARDED across the three GPUs (layer l's pool lives on GPU l%3), so the
    // ~3.5 GB F32 pool never sits on one card and the idle VRAM of the two helper GPUs is put to work.
    static constexpr int kNgpu = 3;
    float *rk = nullptr, *rv = nullptr;      // [n_sparse][cap][HKV*HD]
    float *rik = nullptr;                    // [n_sparse][cap][idx_dim]
    float *pool_k[kNgpu] = { nullptr }, *pool_v[kNgpu] = { nullptr };  // [layers_per_gpu][kPoolBlocks][blk*HKV*HD]
    cudaStream_t pool_st[kNgpu] = { nullptr };
    float *q_relay = nullptr, *da_relay = nullptr;   // pinned: GPU0 <-> pool GPU
    float *pool_q[kNgpu] = { nullptr }, *pool_da[kNgpu] = { nullptr };  // per-pool-GPU q/da (H*HD floats)
    unsigned char *dsel[kNgpu] = { nullptr };        // per-pool-GPU selection mask
    int *pool_blk = nullptr, *pool_age = nullptr;  // host LRU state [n_sparse][kPoolBlocks]
    int n_sparse = 0, cap = 0, n_pos = 0, age = 0, layers_per_gpu = 0;
    bool ready = false;
    int gpu_of(int sparse_idx) const { return sparse_idx % kNgpu; }
    int local_of(int sparse_idx) const { return sparse_idx / kNgpu; }
    bool init(int ns, int cap_, int HKV, int HD, int idx_dim, int blk, int qhd, std::string& err) {
        n_sparse = ns; cap = cap_; n_pos = 0;
        layers_per_gpu = (ns + kNgpu - 1) / kNgpu;
        const size_t kvb = (size_t) ns * cap * HKV * HD;      // F32 elements
        const size_t ikb = (size_t) ns * cap * idx_dim;       // F32 elements
        if (cudaHostAlloc(&rk, kvb * 4, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&rv, kvb * 4, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&rik, ikb * 4, cudaHostAllocDefault) != cudaSuccess) { err = "ram kv pinned"; return false; }
        const size_t pob = (size_t) layers_per_gpu * kPoolBlocks * blk * HKV * HD;   // F32 elements per GPU
        const int smem_bytes = (kPoolBlocks * blk * 2 + 8) * (int) sizeof(float);
        for (int g = 0; g < kNgpu; ++g) {
            cudaSetDevice(g);   // each pool lives on its own card; the smem opt-in is per-context
            if (cudaFuncSetAttribute((const void*) m3k::k_msa_attn_pool, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess ||
                cudaStreamCreate(&pool_st[g]) != cudaSuccess ||
                cudaMalloc(&pool_k[g], pob * 4) != cudaSuccess ||
                cudaMalloc(&pool_v[g], pob * 4) != cudaSuccess ||
                cudaMalloc(&dsel[g], (size_t) 4 * kPoolBlocks) != cudaSuccess) { err = "ram kv pool"; return false; }
        }
        const size_t qbytes = (size_t) qhd * 4;
        if (cudaHostAlloc(&q_relay, qbytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&da_relay, qbytes, cudaHostAllocDefault) != cudaSuccess) { err = "ram kv relay"; return false; }
        for (int g = 0; g < kNgpu; ++g) {
            cudaSetDevice(g);
            if (cudaMalloc(&pool_q[g], qbytes) != cudaSuccess || cudaMalloc(&pool_da[g], qbytes) != cudaSuccess) { err = "ram kv q/da"; return false; }
        }
        cudaSetDevice(0);
        pool_blk = new int[ns * kPoolBlocks];
        pool_age = new int[ns * kPoolBlocks];
        for (int i = 0; i < ns * kPoolBlocks; ++i) { pool_blk[i] = -1; pool_age[i] = 0; }
        ready = true;
        return true;
    }
    void free_() {
        if (rk) { cudaFreeHost(rk); rk = rv = nullptr; }
        if (rik) { cudaFreeHost(rik); rik = nullptr; }
        for (int g = 0; g < kNgpu; ++g) {
            if (pool_k[g]) { cudaFree(pool_k[g]); cudaFree(pool_v[g]); pool_k[g] = pool_v[g] = nullptr; }
            if (dsel[g]) { cudaFree(dsel[g]); dsel[g] = nullptr; }
            if (pool_q[g]) { cudaFree(pool_q[g]); cudaFree(pool_da[g]); pool_q[g] = pool_da[g] = nullptr; }
            if (pool_st[g]) { cudaStreamDestroy(pool_st[g]); pool_st[g] = nullptr; }
        }
        if (q_relay) { cudaFreeHost(q_relay); cudaFreeHost(da_relay); q_relay = da_relay = nullptr; }
        delete[] pool_blk; pool_blk = pool_age = nullptr;
        ready = false; n_pos = 0;
    }
};

/// The GPU forward: the same recipe as forward(), with the projections on the quantised GEMM (iq_mmvq on Q8_1
/// activations) and the elementwise / attention work on the m3k kernels. Experts are streamed from the mmap (their
/// quantised bytes) to the device one at a time - the 95.9 GiB expert stack does not fit in VRAM. Must produce the
/// same token as the CPU forward.
bool forward_gpu(Model& m, const std::vector<int>& tokens, int n_layers, bool use_quant, int expert_cache,
                 int vexp_slots, std::vector<float>& logits, std::string& err) {
    const int T = (int) tokens.size();
    const int nL = n_layers > 0 && n_layers < (int) m.geom.n_layers ? n_layers : (int) m.geom.n_layers;
    const int NE = m.dims.n_embd, H = m.dims.n_head, HKV = m.dims.n_head_kv, HD = m.dims.head_dim;
    const int NFF = (int) m.geom.n_ff;
    const int NFFD = (int) m.geom.n_ff_dense;   // the dense-lead FFN is wider than the expert FFN
    // the persistent F32 expert LRU (opt-in via --experts N); idempotent, so the warm cache survives across forwards
    if (expert_cache > 0) m.init_expert_cache((size_t) expert_cache, NE, NFF);
    // the VRAM expert LRU (opt-in via --vexperts N on the quant path); 32 MB per slot covers this model's experts
    // (~13 MB) with room for the file's widest type
    if (use_quant && vexp_slots > 0 && !g_vexp.init((size_t) vexp_slots, 32u << 20)) {
        err = "vexp arena allocation failed"; return false;
    }
    // stage the model in pinned RAM for fast expert H2D (one-time; M3_NOSTAGE=1 disables)
    if (use_quant && !g_pinned_model.ready && getenv("M3_NOSTAGE") == nullptr) {
        const auto ts = std::chrono::steady_clock::now();
        std::string serr;
        if (m.stage_pinned(serr)) std::fprintf(stderr, "pinned stage  model staged in %.1f s\n", secs(ts));
        else std::fprintf(stderr, "pinned stage  disabled (%s); experts stream from the mmap\n", serr.c_str());
        std::fflush(stderr);
    }
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
        !alloc(&do_, (size_t) NE) || !alloc(&dgate, (size_t) std::max(NFF, NFFD)) || !alloc(&dup, (size_t) std::max(NFF, NFFD)) ||
        !alloc(&dhh, (size_t) std::max(NFF, NFFD)) || !alloc(&df, (size_t) NE) || !alloc(&dr, 128) ||
        !alloc(&dout, (size_t) m.geom.n_vocab) || !alloc(&dkv_k, (size_t) HKV * (T + 8) * HD) ||
        !alloc(&dkv_v, (size_t) HKV * (T + 8) * HD) ||
        !alloc(&diq, (size_t) m.idx.n_groups * m.idx.dim) || !alloc(&dik, (size_t) m.idx.dim) ||
        !alloc(&dikc, (size_t) (T + 8) * m.idx.dim) || !alloc(&dq2, (size_t) H * HD) || !alloc(&dk2, (size_t) HKV * HD) ||
        cudaMalloc(&dmask, (size_t) m.idx.n_groups * (T + 8)) != cudaSuccess) { err = "cudaMalloc"; return false; }
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
    // a projection GEMM: F32 weight -> the plain F32 matvec; quantised weight -> iq_mmvq_f32 on the FLOAT
    // activation (iq_mmvq's Q8_1 activation is not accurate over 60 MoE layers - it compounds to the wrong token)
    // M3_Q81=1: test the Q8_1-activation path (the original design) against the float-activation default
    static const bool m3_q81 = getenv("M3_Q81") != nullptr;
    auto mmf = [&](WProj& w, const float* f32a, void* q81a, float* y, int n_in, int n_out) -> bool {
        if (w.type == (int64_t) Q::Type::F32) {
            m3k::k_f32_mm<<<(n_out + 127) / 128, 128, 0, st>>>((const float*) w.d, f32a, y, n_in, n_out);
            return !die("mmf");
        }
        if (m3_q81) {
            try { K::native_quantize_q8_1(f32a, q81a, n_in, 1, st); }
            catch (const std::exception& e) { err = std::string("quant81: ") + e.what(); return false; }
            K::iq_mmvq((int) w.type, w.d, q81a, y, n_in, n_out, 1, st);
            return !die("mmf q81");
        }
        K::iq_mmvq_f32((int) w.type, w.d, f32a, y, n_in, n_out, st);
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
                std::vector<unsigned char> mask((size_t) m.idx.n_groups * npos, 0);   // per GQA group, NOT the union
                std::vector<float> scores((size_t) nblk);
                for (int g = 0; g < m.idx.n_groups; ++g) {
                    mm::block_scores(diqh.data() + (size_t) g * m.idx.dim, ikc.data(), nullptr, npos, m.idx.dim, m.idx.blk, t, m.idx.local, scores.data());
                    const mm::BlockSel sel = mm::select_blocks(scores.data(), nblk, m.idx.topk);
                    for (int j = 0; j < sel.n; ++j)
                        for (int pp = sel.blocks[j] * m.idx.blk; pp < (sel.blocks[j] + 1) * m.idx.blk && pp < npos; ++pp) mask[(size_t) g * npos + pp] = 1;
                }
                cudaMemcpyAsync(dmask, mask.data(), (size_t) m.idx.n_groups * npos, cudaMemcpyHostToDevice, st);
                m3k::k_msa_attn<<<H, HD, 0, st>>>(dq2, dkv_k, dkv_v, dmask, da, H, HKV, HD, npos, cap);
            } else {
                m3k::k_attn<<<H, HD, 0, st>>>(dq2, dkv_k, dkv_v, da, H, HKV, HD, t + 1, cap, t);
            }
            if (t == T - 1)
if (getenv("M3_DBG_DA") && l == 3) { std::vector<float> d((size_t) H * HD); cudaMemcpyAsync(d.data(), da, (size_t) H * HD * 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); FILE* f = fopen("/tmp/da3_fwd.txt", "w"); for (int i = 0; i < (int) d.size(); ++i) fprintf(f, "%.6f\n", d[i]); fclose(f); }
            if (die("attn")) return false;
            if (g_dump_da && l == 0 && t == T - 1) { g_dump_da_data.resize((size_t) H * HD); cudaMemcpyAsync(g_dump_da_data.data(), da, (size_t) H * HD * sizeof(float), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); }
            // output projection
            if (!mmf(wo, da, hq8, do_, H * HD, NE)) return false;
            m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, do_, NE);
            if (t == T - 1)
if (getenv("M3_DBG_XT") && t == T - 1) { std::vector<float> d((size_t) NE); cudaMemcpyAsync(d.data(), xt, (size_t) NE * 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); FILE* f = fopen((std::string("/tmp/xt_") + std::to_string(l) + "_fwd.txt").c_str(), "w"); for (int i2 = 0; i2 < (int) d.size(); ++i2) fprintf(f, "%.6f\n", d[i2]); fclose(f); }
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
                    if (!mmf(dg, dh, hq8, dgate, NE, NFFD)) return false;
                if (!mmf(du, dh, hq8, dup, NE, NFFD)) return false;
                m3k::k_swiglu_oai<<<(NFFD + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFFD);
                if (!mmf(dd, dhh, hh8, df, NFFD, NE)) return false;
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
            // all-F32 path: one expert's dequantised F32 matrices (~226 MB each); quant path: one expert's raw
            // quantised bytes (~4-12 MB each), dequantised on the GPU by iq_mmvq_f32
            void *exp_g = nullptr, *exp_u = nullptr, *exp_d = nullptr;
            if (use_quant) {
                const size_t qgu = (size_t) NFF * 4096, qd = (size_t) NE * 4096;   // room for any type's row bytes
                if (cudaMalloc(&exp_g, qgu) != cudaSuccess || cudaMalloc(&exp_u, qgu) != cudaSuccess ||
                    cudaMalloc(&exp_d, qd) != cudaSuccess) { err = "cudaMalloc expert pool (quant)"; return false; }
            } else {
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
                    if (use_quant && vexp_slots > 0) {
                        // VRAM LRU: on a hit the expert's quantised bytes are already on the GPU (no PCIe);
                        // on a miss they are streamed in through the pinned staging buffer
                        const uint8_t *eg = nullptr, *eu = nullptr, *ed = nullptr;
                        int gty = 0, uty = 0, dty = 0;
                        if (!g_vexp.get(m, l, r.experts[s], st, &eg, &gty, &eu, &uty, &ed, &dty, err)) {
                            err = "vexp expert " + std::to_string(r.experts[s]) + ": " + err; return false;
                        }
                        K::iq_mmvq_f32(gty, eg, dh, dgate, NE, NFF, st); if (die("vexp gate")) return false;
                        K::iq_mmvq_f32(uty, eu, dh, dup, NE, NFF, st); if (die("vexp up")) return false;
                        m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFF);
                        K::iq_mmvq_f32(dty, ed, dhh, do_, NFF, NE, st); if (die("vexp down")) return false;
                        m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, do_, r.weights[s], NE);
                        if (die("vexp")) return false;
                    } else if (use_quant) {
                        // stream the expert's raw quantised bytes to the pool (a few MB, ~15x less than the F32
                        // matrices) and run the quant-weight x float-activation GEMV on the GPU.  raw_slot_fast
                        // sources the bytes from pinned RAM when staged (faster H2D).
                        int64_t gty, uty, dty; uint64_t gb, ub, db;
                        const uint8_t *gh = m.raw_slot_fast(l, r.experts[s], "ffn_gate_exps.weight", gty, gb, err),
                                      *uh = m.raw_slot_fast(l, r.experts[s], "ffn_up_exps.weight", uty, ub, err),
                                      *dhb = m.raw_slot_fast(l, r.experts[s], "ffn_down_exps.weight", dty, db, err);
                        if (!gh || !uh || !dhb) { err = "expert " + std::to_string(r.experts[s]) + ": " + err; return false; }
                        cudaMemcpyAsync(exp_g, gh, (size_t) gb, cudaMemcpyHostToDevice, st);
                        cudaMemcpyAsync(exp_u, uh, (size_t) ub, cudaMemcpyHostToDevice, st);
                        cudaMemcpyAsync(exp_d, dhb, (size_t) db, cudaMemcpyHostToDevice, st);
                        K::iq_mmvq_f32((int) gty, exp_g, dh, dgate, NE, NFF, st); if (die("expert gate")) return false;
                        K::iq_mmvq_f32((int) uty, exp_u, dh, dup, NE, NFF, st); if (die("expert up")) return false;
                        m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFF);
                        K::iq_mmvq_f32((int) dty, exp_d, dhh, do_, NFF, NE, st); if (die("expert down")) return false;
                        m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, do_, r.weights[s], NE);
                        if (die("expert")) return false;
                    } else {
                        // the expert's three matrices to F32, streamed to the pool.  With --experts N this is the
                        // persistent LRU (dequantised once, then the cached slot is re-streamed over PCIe); without
                        // it, a fresh dequant per use.  Sync first so a prior async copy out of a slot is done before
                        // this call may evict and overwrite it.
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
                        m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, do_, r.weights[s], NE);
                        if (die("expert")) return false;
                    }
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

// ===========================================================================
// Multi-GPU expert runner: the routed experts of a sparse layer are split across the GPUs by
// expert id (expert e runs on GPU e % n_gpus).  Each extra GPU streams its share of the expert
// bytes from pinned RAM over its OWN PCIe link, in parallel with GPU0's attention/router/shared
// work - the single-GPU bottleneck is one link carrying all ~3 GB/token of expert bytes.
// Activations and partial outputs cross via small pinned host relay buffers (24 KB each).
// ===========================================================================
struct M3MultiGpu {
    int n_gpus = 1;                 // GPUs used (1 = single-GPU mode)
    struct Extra {
        int dev = -1;
        void *pool_g = nullptr, *pool_u = nullptr, *pool_d = nullptr;   // one expert's quantised bytes
        float *dgate = nullptr, *dup = nullptr, *dhh = nullptr, *do_ = nullptr, *df = nullptr, *h = nullptr;
        float *dgate_b = nullptr, *dup_b = nullptr, *dhh_b = nullptr, *do_b = nullptr;   // [4][...] grouped-GEMV scratch
        cudaStream_t st = nullptr;
        VramExpertCache cache;      // this GPU's VRAM expert LRU (holds its own id % n_gpus share)
    };
    std::vector<Extra> extra;       // extra[i] = GPU i+1
    float *relay_h = nullptr;       // pinned: activation GPU0 -> extras (NE floats)
    float *relay_out = nullptr;     // pinned: partial outputs extras -> GPU0 ((n-1)*NE floats)
    bool ready = false;

    static int gpu_of(int expert, int n_gpus) { return n_gpus <= 1 ? 0 : (expert % n_gpus); }
};
static M3MultiGpu g_mg;

bool m3_multigpu_init(int NE, int NFF, bool use_quant, int n_gpus, int expslots, std::string& err) {
    if (g_mg.ready) return true;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { err = "cudaGetDeviceCount"; return false; }
    if (n_gpus > devices) n_gpus = devices;
    g_mg.n_gpus = n_gpus;
    if (n_gpus <= 1) return true;   // single-GPU mode: nothing to set up
    const size_t need_gu = use_quant ? (size_t) NFF * 4096 : (size_t) NFF * (size_t) NE * sizeof(float);
    const size_t need_d = use_quant ? (size_t) NE * 4096 : (size_t) NE * (size_t) NFF * sizeof(float);
    g_mg.extra.resize(n_gpus - 1);
    for (int i = 0; i < n_gpus - 1; ++i) {
        const int dev = i + 1;
        cudaSetDevice(dev);
        auto& E = g_mg.extra[i];
        E.dev = dev;
        if (cudaStreamCreate(&E.st) != cudaSuccess) { err = "extra stream"; return false; }
        if (cudaMalloc(&E.pool_g, need_gu) != cudaSuccess || cudaMalloc(&E.pool_u, need_gu) != cudaSuccess ||
            cudaMalloc(&E.pool_d, need_d) != cudaSuccess ||
            cudaMalloc(&E.dgate, (size_t) NFF * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.dup, (size_t) NFF * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.dhh, (size_t) NFF * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.do_, (size_t) NE * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.df, (size_t) NE * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.h, (size_t) NE * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.dgate_b, (size_t) 4 * NFF * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.dup_b, (size_t) 4 * NFF * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.dhh_b, (size_t) 4 * NFF * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&E.do_b, (size_t) 4 * NE * sizeof(float)) != cudaSuccess) {
            err = "extra GPU allocation (device " + std::to_string(dev) + ")"; return false;
        }
        // this GPU's expert LRU: the extra GPUs hold no resident weights, so most of their VRAM is cache.
        // 20 MB per slot fits this model's largest expert (16.4 MB, IQ2_XS gate/up/down) without the 2.4x
        // waste of a 32 MB slot, so the same VRAM holds ~60% more cached instances.
        if (!E.cache.init((size_t) expslots, 32u << 20, dev)) {
            err = "extra GPU expert cache (device " + std::to_string(dev) + ")"; return false;
        }
    }
    cudaSetDevice(0);
    if (cudaHostAlloc(&g_mg.relay_h, (size_t) NE * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&g_mg.relay_out, (size_t) (n_gpus - 1) * NE * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
        err = "relay allocation"; return false;
    }
    g_mg.ready = true;
    return true;
}

// ===========================================================================
// Decode engine: resident projection weights + a persistent KV cache + single-token steps.
//
// The batched forward re-uploads ~7.5 GB of projection weights on every call and recomputes the whole prefix
// for every generated token (the serve loop's 14.9 s/token). The decode engine uploads the weights once,
// appends each new token's k/v to a persistent per-layer cache, and processes exactly one token per step -
// which is what "tokens per second" actually measures. A prompt is just decode_step() called once per prompt
// token in order; the KV cache grows as they are processed.
// ===========================================================================
struct M3DecodeState {
    struct QW { void* d = nullptr; int64_t type = 0; size_t bytes = 0; };
    struct Layer {
        QW wq, wk, wv, wo, sg, su, sd, dg, du, dd;   // dg/du/dd only on dense layers
        float *router = nullptr, *bias = nullptr;    // F32, sparse layers
        float *idxq = nullptr, *idxk = nullptr;      // F32 indexer projections, sparse layers
        float *an = nullptr, *qn = nullptr, *kn = nullptr, *fn = nullptr;
        float *idxqn = nullptr, *idxkn = nullptr;
    };
    std::vector<Layer> layers;
    void* detab = nullptr; int64_t etype = 0; size_t erow = 0;   // embedding table (quantised bytes, device)
    QW head;
    float* out_norm = nullptr;
    float *dkv_k = nullptr, *dkv_v = nullptr;   // dense-lead layers only: [dense_lead][HKV][cap][HD] each
    float* didxc = nullptr;                      // unused in long-ctx mode (indexer keys live in ram.rik)
    float* d_part = nullptr;                     // multi-GPU: one extra GPU's partial output (NE floats, GPU0)
    RamKv ram;                                   // sparse-layer KV + indexer keys in pinned RAM, per-layer GPU block pool
    int cap = 0, n_pos = 0;
    int NE = 0, n_head = 0, HKV = 0, HD = 0, NFF = 0, NFFD = 0, nL = 0, dense_lead = 0, n_sparse = 0;
    int idx_dim = 0, idx_groups = 0, idx_blk = 0, idx_topk = 0, idx_local = 0;
    // Step scratch, allocated ONCE at init (not per step).  A per-step cudaFree is a whole-device synchronous
    // drain, so reading a 198-token prompt used to do ~198x18 implicit drains that destroyed the async overlap
    // of the three GPUs' expert streams.  The persistent stream keeps ordering (FIFO) with no per-step sync; the
    // cross-GPU relay handoffs carry their own explicit cudaStreamSynchronize calls.
    struct Scratch {
        float *dx=nullptr, *dh=nullptr, *dq=nullptr, *dk=nullptr, *dv=nullptr, *da=nullptr, *do_=nullptr,
              *dgate=nullptr, *dup=nullptr, *dhh=nullptr, *df=nullptr, *dout=nullptr,
              *diq=nullptr, *dik=nullptr, *dq2=nullptr, *dk2=nullptr;
        unsigned char* dmask=nullptr;
        int32_t* dtok=nullptr;
        cudaStream_t st=nullptr;
    } scratch;
    // Chunked-prefill scratch (C = chunk rows), allocated once in m3_decode_init.
    struct Chunk {
        float *x=nullptr, *h=nullptr, *q=nullptr, *k=nullptr, *v=nullptr, *a=nullptr, *o=nullptr;
        float *iq=nullptr, *ik=nullptr, *q2=nullptr, *k2=nullptr, *ckv_k=nullptr, *ckv_v=nullptr;
        float *gate=nullptr, *up=nullptr, *hh=nullptr, *df=nullptr, *dout=nullptr, *score=nullptr, *idxs=nullptr;
        float *bnc[4] = {nullptr, nullptr, nullptr, nullptr};
        int bnc_i = 0;
        int *rows=nullptr;
        int C = 0;
    } chunk;
    bool ready = false;
};
// One per process: initialised at startup (before READY) in serve mode, so "ready" means ready to serve
// immediately and the first request does not block on the one-time model staging (which would trip the
// request's silence watchdog).
static M3DecodeState g_ds;

void m3_decode_free(M3DecodeState& ds) {
    for (auto& L : ds.layers) {
        for (auto* p : { &L.wq.d, &L.wk.d, &L.wv.d, &L.wo.d, &L.sg.d, &L.su.d, &L.sd.d, &L.dg.d, &L.du.d, &L.dd.d })
            if (*p) { cudaFree(*p); *p = nullptr; }
        for (auto* p : { &L.router, &L.bias, &L.idxq, &L.idxk, &L.an, &L.qn, &L.kn, &L.fn, &L.idxqn, &L.idxkn })
            if (*p) { cudaFree(*p); *p = nullptr; }
    }
    ds.layers.clear();
    if (ds.detab) { cudaFree(ds.detab); ds.detab = nullptr; }
    if (ds.head.d) { cudaFree(ds.head.d); ds.head.d = nullptr; }
    if (ds.out_norm) { cudaFree(ds.out_norm); ds.out_norm = nullptr; }
    if (ds.dkv_k) { cudaFree(ds.dkv_k); ds.dkv_k = nullptr; }
    if (ds.dkv_v) { cudaFree(ds.dkv_v); ds.dkv_v = nullptr; }
    if (ds.didxc) { cudaFree(ds.didxc); ds.didxc = nullptr; }
    ds.ram.free_();
    if (ds.d_part) { cudaFree(ds.d_part); ds.d_part = nullptr; }
    {
        auto& S = ds.scratch;
        cudaFree(S.dx); cudaFree(S.dh); cudaFree(S.dq); cudaFree(S.dk); cudaFree(S.dv); cudaFree(S.da);
        cudaFree(S.do_); cudaFree(S.dgate); cudaFree(S.dup); cudaFree(S.dhh); cudaFree(S.df); cudaFree(S.dout);
        cudaFree(S.diq); cudaFree(S.dik); cudaFree(S.dq2); cudaFree(S.dk2); cudaFree(S.dmask); cudaFree(S.dtok);
        if (S.st) { cudaStreamDestroy(S.st); S.st = nullptr; }
        S.dx = S.dh = S.dq = S.dk = S.dv = S.da = S.do_ = nullptr;
        S.dgate = S.dup = S.dhh = S.df = S.dout = S.diq = S.dik = S.dq2 = S.dk2 = nullptr;
        S.dmask = nullptr; S.dtok = nullptr;
    }
    ds.n_pos = 0; ds.ready = false;
}

bool m3_decode_init(Model& m, M3DecodeState& ds, bool use_quant, int cap, int n_gpus, int expslots, std::string& err) {
    if (ds.ready) m3_decode_free(ds);
    // multi-GPU expert runner (extra GPUs compute their share of the routed experts in parallel)
    const int NE0 = m.dims.n_embd, NFF0 = (int) m.geom.n_ff;
    if (use_quant && !m3_multigpu_init(NE0, NFF0, true, n_gpus, expslots, err)) return false;
    // stage the model in pinned RAM so expert H2D runs at the pinned rate (one-time, ~file-size / CPU-copy
    // speed); M3_NOSTAGE=1 disables it (experts then stream from the pageable mmap)
    if (use_quant && !g_pinned_model.ready && getenv("M3_NOSTAGE") == nullptr) {
        const auto ts = std::chrono::steady_clock::now();
        std::string serr;
        if (m.stage_pinned(serr)) std::fprintf(stderr, "pinned stage  model staged in %.1f s\n", secs(ts));
        else std::fprintf(stderr, "pinned stage  disabled (%s); experts stream from the mmap\n", serr.c_str());
        std::fflush(stderr);
    }
    const int NE = m.dims.n_embd, H = m.dims.n_head, HKV = m.dims.n_head_kv, HD = m.dims.head_dim;
    const int NFF = (int) m.geom.n_ff;
    const int NFFD = (int) m.geom.n_ff_dense;   // the dense-lead FFN is wider than the expert FFN
    const int nL = (int) m.geom.n_layers;
    ds.NE = NE; ds.n_head = H; ds.HKV = HKV; ds.HD = HD; ds.NFF = NFF; ds.NFFD = NFFD; ds.nL = nL;
    ds.dense_lead = (int) m.geom.n_dense_lead;
    ds.n_sparse = nL - ds.dense_lead;
    ds.cap = cap; ds.n_pos = 0;
    ds.idx_dim = m.idx.dim; ds.idx_groups = m.idx.n_groups; ds.idx_blk = m.idx.blk;
    ds.idx_topk = m.idx.topk; ds.idx_local = m.idx.local;

    auto upload_quant = [&](const std::string& name, M3DecodeState::QW& w) -> bool {
        int64_t ty; std::vector<uint64_t> sh;
        const uint8_t* h = m.raw(name, ty, sh, err);
        if (!h) return false;
        const int64_t n = sh.size() >= 2 ? (int64_t) sh[0] * sh[1] : 0;
        if (use_quant && (Q::Type) ty != Q::Type::F32) {
            w.type = ty; w.bytes = Q::row_bytes((int) ty, n);
            if (w.bytes == 0) { err = "no row bytes for " + name; return false; }
            if (cudaMalloc(&w.d, w.bytes) != cudaSuccess) { err = "cudaMalloc " + name; return false; }
            if (cudaMemcpy(w.d, h, w.bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "H2D " + name; return false; }
            return true;
        }
        std::vector<float> f((size_t) n);
        if ((Q::Type) ty == Q::Type::F32) std::memcpy(f.data(), h, (size_t) n * sizeof(float));
        else if (!Q::dequant_row((Q::Type) ty, h, f.data(), n)) { err = "dequant " + name; return false; }
        w.type = (int64_t) Q::Type::F32; w.bytes = (size_t) n * sizeof(float);
        if (cudaMalloc(&w.d, w.bytes) != cudaSuccess) { err = "cudaMalloc " + name; return false; }
        cudaMemcpy(w.d, f.data(), w.bytes, cudaMemcpyHostToDevice);
        return true;
    };
    auto upload_f32 = [&](const std::string& name, size_t n, float** p) -> bool {
        int64_t ty; std::vector<uint64_t> sh;
        const uint8_t* h = m.raw(name, ty, sh, err);
        if (!h) return false;
        std::vector<float> f(n);
        if ((Q::Type) ty == Q::Type::F32) std::memcpy(f.data(), h, n * sizeof(float));
        else if (!Q::dequant_row((Q::Type) ty, h, f.data(), (int64_t) n)) { err = "dequant " + name; return false; }
        if (cudaMalloc(p, n * sizeof(float)) != cudaSuccess) { err = "cudaMalloc " + name; return false; }
        cudaMemcpy(*p, f.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        return true;
    };

    ds.layers.resize(nL);
    for (int l = 0; l < nL; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        auto& L = ds.layers[l];
        if (!upload_quant(p + "attn_q.weight", L.wq) || !upload_quant(p + "attn_k.weight", L.wk) ||
            !upload_quant(p + "attn_v.weight", L.wv) || !upload_quant(p + "attn_output.weight", L.wo)) { err = "layer " + std::to_string(l) + " attn weights"; return false; }
        if (!upload_f32(p + "attn_norm.weight", NE, &L.an) || !upload_f32(p + "attn_q_norm.weight", HD, &L.qn) ||
            !upload_f32(p + "attn_k_norm.weight", HD, &L.kn) || !upload_f32(p + "ffn_norm.weight", NE, &L.fn)) {
            err = "layer " + std::to_string(l) + " norms"; return false;
        }
        if (l < ds.dense_lead) {
            if (!upload_quant(p + "ffn_gate.weight", L.dg) || !upload_quant(p + "ffn_up.weight", L.du) ||
                !upload_quant(p + "ffn_down.weight", L.dd)) { err = "layer " + std::to_string(l) + " dense ffn"; return false; }
        } else {
            if (!upload_quant(p + "ffn_gate_shexp.weight", L.sg) || !upload_quant(p + "ffn_up_shexp.weight", L.su) ||
                !upload_quant(p + "ffn_down_shexp.weight", L.sd)) { err = "layer " + std::to_string(l) + " shared expert"; return false; }
            if (!upload_f32(p + "ffn_gate_inp.weight", (size_t) NE * m.geom.n_expert, &L.router) ||
                !upload_f32(p + "exp_probs_b.bias", (size_t) m.geom.n_expert, &L.bias) ||
                !upload_f32(p + "indexer.q_proj.weight", (size_t) NE * (m.idx.n_groups * m.idx.dim), &L.idxq) ||
                !upload_f32(p + "indexer.k_proj.weight", (size_t) NE * m.idx.dim, &L.idxk) ||
                !upload_f32(p + "indexer.q_norm.weight", (size_t) m.idx.dim, &L.idxqn) ||
                !upload_f32(p + "indexer.k_norm.weight", (size_t) m.idx.dim, &L.idxkn)) {
                err = "layer " + std::to_string(l) + " router/indexer"; return false;
            }
        }
    }
    {
        int64_t ty; std::vector<uint64_t> sh;
        const uint8_t* h = m.raw("token_embd.weight", ty, sh, err);
        if (!h) { err = "token_embd.weight"; return false; }
        ds.etype = ty; ds.erow = Q::row_bytes((int) ty, NE);
        const size_t total = ds.erow * (size_t) m.geom.n_vocab;
        if (cudaMalloc(&ds.detab, total) != cudaSuccess ||
            cudaMemcpy(ds.detab, h, total, cudaMemcpyHostToDevice) != cudaSuccess) { err = "embedding table"; return false; }
    }
    if (!upload_quant("output.weight", ds.head)) { err = "output.weight"; return false; }
    if (!upload_f32("output_norm.weight", NE, &ds.out_norm)) { err = "output_norm.weight"; return false; }
    // dense-lead layers keep their full KV on the GPU; the sparse layers' KV + indexer keys go to pinned RAM
    // (a 250k-context F32 sparse KV would be ~65 GB, long past any card here)
    if (cudaMalloc(&ds.dkv_k, (size_t) ds.dense_lead * HKV * cap * HD * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ds.dkv_v, (size_t) ds.dense_lead * HKV * cap * HD * sizeof(float)) != cudaSuccess) { err = "dense KV allocation: " + std::string(cudaGetErrorString(cudaGetLastError())); return false; }
    if (!ds.ram.init(ds.n_sparse, cap, HKV, HD, m.idx.dim, m.idx.blk, H * HD, err)) { err = "ram kv: " + err; return false; }
    if (cudaMalloc(&ds.d_part, (size_t) NE * sizeof(float)) != cudaSuccess) { err = "d_part allocation"; return false; }
    {   // chunked-prefill scratch: C rows of every per-position buffer
        const int C = 256;
        const size_t NEb = (size_t) NE * 4, HDb = (size_t) H * HD * 4, KHDb = (size_t) HKV * HD * 4,
                     NFFb = (size_t) NFF * 4, NFFDb = (size_t) NFFD * 4,
                     IDb = (size_t) m.idx.n_groups * m.idx.dim * 4, ID1b = (size_t) m.idx.dim * 4;
        auto cm = [&](float** p, size_t bytes) { if (cudaMalloc(p, bytes) != cudaSuccess) return false; return true; };
        auto& c = ds.chunk; c.C = C;
        if (!cm(&c.x, C*NEb) || !cm(&c.h, C*NEb) || !cm(&c.q, C*HDb) || !cm(&c.k, C*KHDb) || !cm(&c.v, C*KHDb) ||
            !cm(&c.a, C*HDb) || !cm(&c.o, C*NEb) || !cm(&c.iq, C*IDb) || !cm(&c.ik, C*ID1b) ||
            !cm(&c.q2, C*HDb) || !cm(&c.k2, C*KHDb) || !cm(&c.ckv_k, C*KHDb) || !cm(&c.ckv_v, C*KHDb) ||
            !cm(&c.gate, C*NFFb) || !cm(&c.up, C*NFFb) || !cm(&c.hh, C*NFFb) || !cm(&c.df, C*NEb) ||
            !cm(&c.dout, C*NEb) || !cm(&c.score, C*(size_t) m.geom.n_expert * 4) ||
            !cm(&c.idxs, (size_t) C * 8192 * 4) ||
            !cm(&c.bnc[0], C*NFFDb) || !cm(&c.bnc[1], C*NFFDb) || !cm(&c.bnc[2], C*NFFDb) || !cm(&c.bnc[3], C*NFFDb) ||
            cudaMalloc(&c.rows, (size_t) C * m.geom.n_expert * 4)) {
            err = "chunk scratch"; return false;
        }
    }
    {
        auto& S = ds.scratch;
        auto A = [&](float** p, size_t n) { return cudaMalloc(p, n * sizeof(float)) == cudaSuccess; };
        if (!A(&S.dx, NE) || !A(&S.dh, NE) || !A(&S.dq, (size_t) H * HD) || !A(&S.dk, (size_t) HKV * HD) ||
            !A(&S.dv, (size_t) HKV * HD) || !A(&S.da, (size_t) H * HD) || !A(&S.do_, NE) ||
            !A(&S.dgate, (size_t) std::max(NFF, NFFD)) || !A(&S.dup, (size_t) std::max(NFF, NFFD)) ||
            !A(&S.dhh, (size_t) std::max(NFF, NFFD)) || !A(&S.df, NE) ||
            !A(&S.dout, (size_t) m.geom.n_vocab) || !A(&S.diq, (size_t) m.idx.n_groups * m.idx.dim) ||
            !A(&S.dik, m.idx.dim) || !A(&S.dq2, (size_t) H * HD) || !A(&S.dk2, (size_t) HKV * HD) ||
            cudaMalloc(&S.dmask, (size_t) m.idx.n_groups * RamKv::kPoolBlocks) != cudaSuccess ||   // per-group pool mask
            cudaMalloc(&S.dtok, sizeof(int32_t)) != cudaSuccess) { err = "decode scratch"; return false; }
        if (cudaStreamCreate(&S.st) != cudaSuccess) { err = "decode stream"; return false; }
    }
    ds.ready = true;
    return true;
}

// One decode step: process `token_id` at position ds.n_pos, attending to 0..P from the persistent cache,
// appending its k/v, and returning the next-token logits. Advances ds.n_pos by one on success.
bool m3_layer_attn(Model& m, M3DecodeState& ds, int l, int P, float* xt, std::string& err) {
    const int NE = ds.NE, H = ds.n_head, HKV = ds.HKV, HD = ds.HD;
    auto& SC = ds.scratch;
    cudaStream_t st = SC.st;
    auto die = [&](const char* w) -> bool { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); return true; } return false; };
    auto mmf = [&](const M3DecodeState::QW& w, const float* f32a, float* y, int n_in, int n_out) -> bool {
        if (w.type == (int64_t) Q::Type::F32) {
            m3k::k_f32_mm<<<(n_out + 127) / 128, 128, 0, st>>>((const float*) w.d, f32a, y, n_in, n_out);
        } else {
            K::iq_mmvq_f32((int) w.type, w.d, f32a, y, n_in, n_out, st);
        }
        return !die("mmf");
    };
    const auto& L = ds.layers[l];
    const bool sparse = l >= ds.dense_lead;
    int sparse_idx = sparse ? l - ds.dense_lead : 0;
    float *dh=SC.dh, *dq=SC.dq, *dk=SC.dk, *dv=SC.dv, *da=SC.da, *do_=SC.do_,
          *diq=SC.diq, *dik=SC.dik, *dq2=SC.dq2, *dk2=SC.dk2;
    const bool tprof = false;
    auto now_ms = []() -> double { return 0.0; };
        // ---- attention ----
        m3k::k_rms<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, L.an, dh, NE, (float) m.geom.rms_eps);
        if (die("rms1")) return false;
        if (!mmf(L.wq, dh, dq, NE, H * HD)) return false;
        if (!mmf(L.wk, dh, dk, NE, HKV * HD)) return false;
        if (!mmf(L.wv, dh, dv, NE, HKV * HD)) return false;
        m3k::k_rms<<<dim3(1, H), HD, 0, st>>>(dq, L.qn, dq2, HD, m.dims.eps);
        m3k::k_rms<<<dim3(1, HKV), HD, 0, st>>>(dk, L.kn, dk2, HD, m.dims.eps);
        m3k::k_rope<<<H + HKV, m.dims.n_rot / 2, 0, st>>>(dq2, dk2, H, HKV, HD, m.dims.n_rot, (float) P, m.dims.rope_base);
        // Dense-lead layers keep their full KV on the GPU; sparse layers read a per-layer block pool backed by
        // pinned RAM (a whole-prefix F32 KV would be ~65 GB at 250k context - past any card here).
        float* lkv_k = ds.dkv_k + (size_t) l * HKV * ds.cap * HD;
        float* lkv_v = ds.dkv_v + (size_t) l * HKV * ds.cap * HD;
        if (!sparse) {
            m3k::k_kv_append<<<HKV, HD, 0, st>>>(lkv_k, dk2, HKV, HD, ds.cap, P);
            m3k::k_kv_append<<<HKV, HD, 0, st>>>(lkv_v, dv, HKV, HD, ds.cap, P);
            m3k::k_attn<<<H, HD, 0, st>>>(dq2, lkv_k, lkv_v, da, H, HKV, HD, P + 1, ds.cap, P);
        } else {
            // MSA indexer: score the key blocks, select top-k per GQA group
            m3k::k_f32_mm<<<(m.idx.n_groups * m.idx.dim + 127) / 128, 128, 0, st>>>(L.idxq, dh, diq, NE, m.idx.n_groups * m.idx.dim);
            m3k::k_f32_mm<<<(m.idx.dim + 127) / 128, 128, 0, st>>>(L.idxk, dh, dik, NE, m.idx.dim);
            m3k::k_rms<<<dim3(1, m.idx.n_groups), m.idx.dim, 0, st>>>(diq, L.idxqn, diq, m.idx.dim, m.dims.eps);
            m3k::k_rms<<<dim3(1, 1), m.idx.dim, 0, st>>>(dik, L.idxkn, dik, m.idx.dim, m.dims.eps);
            m3k::k_rope<<<m.idx.n_groups, m.dims.n_rot / 2, 0, st>>>(diq, nullptr, m.idx.n_groups, 0, m.idx.dim, m.dims.n_rot, (float) P, m.dims.rope_base);
            m3k::k_rope<<<1, m.dims.n_rot / 2, 0, st>>>(dik, nullptr, 1, 0, m.idx.dim, m.dims.n_rot, (float) P, m.dims.rope_base);
            if (die("indexer rope")) return false;
            auto& ram = ds.ram;
            auto ck = [&](const char* w) { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); } return e == cudaSuccess; };
            const double sp0 = tprof ? now_ms() : 0.0;
            double sp_d2h = 0, sp_score = 0, sp_fetch = 0, sp_relay = 0, sp_kern = 0;
            const int ns = m.idx.n_groups, idim = m.idx.dim, blk = m.idx.blk, nb = RamKv::kPoolBlocks;
            const size_t kvn = (size_t) HKV * HD;
            // one small D2H batch: indexer q/k + this position's k/v (replaces the old O(prefix) indexer copy)
            std::vector<float> down((size_t) ns * idim + idim + 2 * kvn);
            cudaMemcpyAsync(down.data(), diq, (size_t) ns * idim * 4, cudaMemcpyDeviceToHost, st);
            cudaMemcpyAsync(down.data() + (size_t) ns * idim, dik, (size_t) idim * 4, cudaMemcpyDeviceToHost, st);
            cudaMemcpyAsync(down.data() + (size_t) ns * idim + idim, dk2, kvn * 4, cudaMemcpyDeviceToHost, st);
            cudaMemcpyAsync(down.data() + (size_t) ns * idim + idim + kvn, dv, kvn * 4, cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            // append position P to the pinned-RAM caches (F16 K/V, F32 indexer key)
            {
                const float* kh = down.data() + (size_t) ns * idim + idim;
                const float* vh = down.data() + (size_t) ns * idim + idim + kvn;
                float* rk_l = ram.rk + ((size_t) sparse_idx * ram.cap + P) * kvn;
                float* rv_l = ram.rv + ((size_t) sparse_idx * ram.cap + P) * kvn;
                std::memcpy(rk_l, kh, kvn * 4);
                std::memcpy(rv_l, vh, kvn * 4);
                std::memcpy(ram.rik + ((size_t) sparse_idx * ram.cap + P) * idim, down.data() + (size_t) ns * idim, (size_t) idim * 4);   // dik, not dk2
            }
            // score blocks against the RAM keys (host side, O(nblk)), select per group, build the union
            const float* rikc = ram.rik + (size_t) sparse_idx * ram.cap * idim;
            const int npos = P + 1, nblk = (npos + blk - 1) / blk;
            std::vector<float> bscore((size_t) nblk);
            std::vector<mm::BlockSel> sels((size_t) ns);
            std::vector<int> union_blk;
            union_blk.reserve((size_t) ns * m.idx.topk);
            {
                std::vector<unsigned char> seen((size_t) nblk, 0);   // block indices, up to nblk-1
                for (int g = 0; g < ns; ++g) {
                    mm::block_scores(down.data() + (size_t) g * idim, rikc, nullptr, npos, idim, blk, P, m.idx.local, bscore.data());
                    sels[(size_t) g] = mm::select_blocks(bscore.data(), nblk, m.idx.topk);
                    for (int j = 0; j < sels[(size_t) g].n; ++j)
                        if (!seen[sels[(size_t) g].blocks[j]]) { seen[sels[(size_t) g].blocks[j]] = 1; union_blk.push_back(sels[(size_t) g].blocks[j]); }
                }
            }
            // make every selected block resident in the per-layer GPU pool; the in-progress block is always
            // refetched (it just gained a position); victims are the least-recently-used blocks NOT in the union
            const int pg = ram.gpu_of(sparse_idx), pl = ram.local_of(sparse_idx);
            cudaStream_t pst = ram.pool_st[pg];
            cudaSetDevice(pg);   // all pool-side work happens with this card current
            int* pblk = ram.pool_blk + (size_t) sparse_idx * nb;
            int* page = ram.pool_age + (size_t) sparse_idx * nb;
            const int b_last = P / blk;
            for (int b : union_blk) {
                int slot = -1;
                for (int s = 0; s < nb; ++s) if (pblk[s] == b) { slot = s; break; }
                bool fresh = false;
                if (slot < 0) {
                    int victim = -1;
                    for (int s = 0; s < nb; ++s) {
                        bool needed = false;
                        for (int u : union_blk) if (pblk[s] == u) { needed = true; break; }
                        if (pblk[s] < 0 || (!needed && (victim < 0 || page[s] < page[victim]))) victim = s;
                    }
                    pblk[victim] = b;
                    slot = victim;
                    fresh = true;
                }
                page[slot] = ++ram.age;
                if (fresh || b == b_last) {
                    // the RAM bytes go straight into the pool slot on this layer's pool GPU
                    const size_t base = (size_t) sparse_idx * ram.cap + (size_t) b * blk;
                    const float* srck = ram.rk + base * kvn;
                    const float* srcv = ram.rv + base * kvn;
                    float* pok = ram.pool_k[pg] + ((size_t) pl * nb + slot) * (size_t) blk * kvn;
                    float* pov = ram.pool_v[pg] + ((size_t) pl * nb + slot) * (size_t) blk * kvn;
                    cudaMemcpyAsync(pok, srck, (size_t) blk * kvn * 4, cudaMemcpyHostToDevice, pst);
                    if (!ck("fetch k H2D")) return false;
                    cudaMemcpyAsync(pov, srcv, (size_t) blk * kvn * 4, cudaMemcpyHostToDevice, pst);
                    if (!ck("fetch v H2D")) return false;
                }
            }
            // per-group pool mask + the in-progress block's valid length (its tail is unwritten padding)
            std::vector<unsigned char> sel_pool((size_t) ns * nb, 0);
            for (int g = 0; g < ns; ++g)
                for (int j = 0; j < sels[(size_t) g].n; ++j)
                    for (int s = 0; s < nb; ++s)
                        if (pblk[s] == sels[(size_t) g].blocks[j]) sel_pool[(size_t) g * nb + s] = 1;
            int last_slot = -1, last_valid = 0;
            for (int s = 0; s < nb; ++s) if (pblk[s] == b_last) { last_slot = s; last_valid = P % blk + 1; break; }
            // relay q to this layer's pool GPU, run the pool attention there, relay the output back.
            // Cross-device sync is host-side (consumer cards have no P2P); the decode path already
            // syncs the host once per sparse layer for the indexer batch.
            const size_t qbytes = (size_t) H * HD * 4;
            cudaMemcpyAsync(ram.q_relay, dq2, qbytes, cudaMemcpyDeviceToHost, st);
            if (cudaStreamSynchronize(st) != cudaSuccess) { err = "q D2H sync"; return false; }
            cudaMemcpyAsync(ram.pool_q[pg], ram.q_relay, qbytes, cudaMemcpyHostToDevice, pst);
            if (!ck("q H2D")) return false;
            cudaMemcpyAsync(ram.dsel[pg], sel_pool.data(), (size_t) ns * nb, cudaMemcpyHostToDevice, pst);
            if (!ck("sel H2D")) return false;
            m3k::k_msa_attn_pool<<<H, HD, (nb * blk * 2 + 8) * (int) sizeof(float), pst>>>(ram.pool_q[pg],
                ram.pool_k[pg] + (size_t) pl * nb * (size_t) blk * kvn,
                ram.pool_v[pg] + (size_t) pl * nb * (size_t) blk * kvn,
                ram.dsel[pg], ram.pool_da[pg], H, HKV, HD, nb, blk, last_slot, last_valid);
            if (!ck("pool launch")) return false;
            cudaMemcpyAsync(ram.da_relay, ram.pool_da[pg], qbytes, cudaMemcpyDeviceToHost, pst);
            if (cudaStreamSynchronize(pst) != cudaSuccess) { err = "pool attn sync"; return false; }
            cudaSetDevice(0);
            cudaMemcpyAsync(da, ram.da_relay, qbytes, cudaMemcpyHostToDevice, st);
            if (!ck("da H2D")) return false;
            ++sparse_idx;
        }
        if (die("attn")) return false;
        if (!mmf(L.wo, da, do_, H * HD, NE)) return false;
        m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, do_, NE);
        if (die("add1")) return false;    return true;
}

bool m3_layer_ffn(Model& m, M3DecodeState& ds, int vexp_slots, int l, float* xt, std::string& err) {
    const int NE = ds.NE, NFF = ds.NFF, NFFD = ds.NFFD;
    auto& SC = ds.scratch;
    cudaStream_t st = SC.st;
    auto die = [&](const char* w) -> bool { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); return true; } return false; };
    auto mmf = [&](const M3DecodeState::QW& w, const float* f32a, float* y, int n_in, int n_out) -> bool {
        if (w.type == (int64_t) Q::Type::F32) {
            m3k::k_f32_mm<<<(n_out + 127) / 128, 128, 0, st>>>((const float*) w.d, f32a, y, n_in, n_out);
        } else {
            K::iq_mmvq_f32((int) w.type, w.d, f32a, y, n_in, n_out, st);
        }
        return !die("mmf");
    };
    const auto& L = ds.layers[l];
    const bool sparse = l >= ds.dense_lead;
    float *dh=SC.dh, *dgate=SC.dgate, *dup=SC.dup, *dhh=SC.dhh, *df=SC.df, *do_=SC.do_;
    std::vector<float> scores((size_t) m.geom.n_expert), bias_host((size_t) m.geom.n_expert);
    const bool tprof = false;
    auto now_ms = []() -> double { return 0.0; };
        // ---- FFN ----
        if (!sparse) {
            m3k::k_rms<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, L.fn, dh, NE, (float) m.geom.rms_eps);
            if (!mmf(L.dg, dh, dgate, NE, NFFD)) return false;
            if (!mmf(L.du, dh, dup, NE, NFFD)) return false;
            m3k::k_swiglu_oai<<<(NFFD + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFFD);
            if (!mmf(L.dd, dhh, df, NFFD, NE)) return false;
            m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, df, NE);
            if (die("dense_ffn")) return false;
        } else {
            const double t_router0 = tprof ? now_ms() : 0.0;
            m3k::k_rms<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, L.fn, dh, NE, (float) m.geom.rms_eps);
            // router on the GPU (resident F32 weights), scores back to the host for the top-k
            float* dr = nullptr;
            if (cudaMalloc(&dr, (size_t) m.geom.n_expert * sizeof(float)) != cudaSuccess) { err = "router scores"; return false; }
            m3k::k_f32_mm<<<(m.geom.n_expert + 127) / 128, 128, 0, st>>>(L.router, dh, dr, NE, (int) m.geom.n_expert);
            cudaMemcpyAsync(scores.data(), dr, (size_t) m.geom.n_expert * sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaMemcpyAsync(bias_host.data(), L.bias, (size_t) m.geom.n_expert * sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            cudaFree(dr);
            if (die("router")) return false;
            const double t_routed0 = tprof ? now_ms() : 0.0;
            mm::MoEWeights mw;
            mw.n_expert = (int) m.geom.n_expert; mw.n_expert_used = (int) m.geom.n_expert_used;
            mw.n_ff = NFF; mw.n_embd = NE; mw.scale = 2.0f; mw.norm = true;
            const mm::Route r = mm::route(scores.data(), bias_host.data(), mw.n_expert, mw.n_expert_used, true, 2.0f);
            // Compute routed expert s (streaming its quantised bytes from pinned RAM / the mmap) and accumulate
            // w_s * expert(h) into `odff`.  Used on every GPU: the buffers + stream are parameters.
            auto compute_expert = [&](int s, void* pg, void* pu, void* pd, const float* hsrc,
                                      float* og, float* ou, float* ohh, float* odo, float* odff,
                                      cudaStream_t ostr, const char* tag) -> bool {
                int64_t gty, uty, dty; uint64_t gb, ub, db;
                const uint8_t *gh = m.raw_slot_fast(l, r.experts[s], "ffn_gate_exps.weight", gty, gb, err),
                              *uh = m.raw_slot_fast(l, r.experts[s], "ffn_up_exps.weight", uty, ub, err),
                              *dhb = m.raw_slot_fast(l, r.experts[s], "ffn_down_exps.weight", dty, db, err);
                if (!gh || !uh || !dhb) { err = "expert " + std::to_string(r.experts[s]) + ": " + err; return false; }
                if (cudaMemcpyAsync(pg, gh, (size_t) gb, cudaMemcpyHostToDevice, ostr) != cudaSuccess ||
                    cudaMemcpyAsync(pu, uh, (size_t) ub, cudaMemcpyHostToDevice, ostr) != cudaSuccess ||
                    cudaMemcpyAsync(pd, dhb, (size_t) db, cudaMemcpyHostToDevice, ostr) != cudaSuccess)
                    return false;
                K::iq_mmvq_f32((int) gty, pg, hsrc, og, NE, NFF, ostr);
                K::iq_mmvq_f32((int) uty, pu, hsrc, ou, NE, NFF, ostr);
                m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, ostr>>>(og, ou, ohh, NFF);
                K::iq_mmvq_f32((int) dty, pd, ohh, odo, NFF, NE, ostr);
                m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, ostr>>>(odff, odo, r.weights[s], NE);
                const cudaError_t ee = cudaGetLastError();
                if (ee != cudaSuccess) { err = std::string(tag) + ": " + cudaGetErrorString(ee); return false; }
                return true;
            };
            if (g_mg.n_gpus > 1) {
                // ---- multi-GPU expert parallelism ----
                // The routed experts are split across the GPUs by id % n_gpus.  Phase A enqueues every extra
                // GPU's work WITHOUT waiting (their PCIe links then run concurrently); phase B computes GPU0's
                // share while the extras' DMA/GEMVs are in flight; phase C waits and folds the partials.
                cudaMemcpyAsync(g_mg.relay_h, dh, (size_t) NE * sizeof(float), cudaMemcpyDeviceToHost, st);
                cudaStreamSynchronize(st);
                // Phase A: enqueue the extra GPUs' work (async)
                for (int i = 0; i < (int) g_mg.extra.size(); ++i) {
                    const int dev = i + 1;
                    cudaSetDevice(dev);
                    auto& E = g_mg.extra[i];
                    cudaMemcpyAsync(E.h, g_mg.relay_h, (size_t) NE * sizeof(float), cudaMemcpyHostToDevice, E.st);
                    cudaMemsetAsync(E.df, 0, (size_t) NE * sizeof(float), E.st);
                    // Grouped: fetch every assigned expert's bytes first (hits stay in VRAM, misses stream)
                    // so all copies queue back-to-back, then one batched GEMV per projection.
                    {
                        const uint8_t *eg[4] = { nullptr }, *eu[4] = { nullptr }, *ed[4] = { nullptr };
                        float wgt[4]; int gty = 0, uty = 0, dty = 0, E_cnt = 0;
                        for (int s = 0; s < r.n; ++s) {
                            if (g_mg.gpu_of(r.experts[s], g_mg.n_gpus) != dev) continue;
                            if (E.cache.n_slots > 0) {
                                if (!E.cache.get(m, l, r.experts[s], E.st, &eg[E_cnt], &gty, &eu[E_cnt], &uty, &ed[E_cnt], &dty, err)) {
                                    err = "extra GPU vexp expert " + std::to_string(r.experts[s]) + ": " + err; cudaSetDevice(0); return false;
                                }
                                wgt[E_cnt] = r.weights[s]; ++E_cnt;
                            } else if (!compute_expert(s, E.pool_g, E.pool_u, E.pool_d, E.h, E.dgate, E.dup, E.dhh, E.do_, E.df, E.st, "extra GPU expert")) {
                                cudaSetDevice(0); return false;
                            }
                        }
                        if (E_cnt > 0) {
                            void* wp[4];
                            for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) eg[e];
                            K::iq_mmvq_f32_batch(E_cnt, wp, gty, NE, NFF, E.h, E.dgate_b, E.st);
                            for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) eu[e];
                            K::iq_mmvq_f32_batch(E_cnt, wp, uty, NE, NFF, E.h, E.dup_b, E.st);
                            for (int e = 0; e < E_cnt; ++e)
                                m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, E.st>>>(E.dgate_b + (size_t) e * NFF, E.dup_b + (size_t) e * NFF, E.dhh_b + (size_t) e * NFF, NFF);
                            for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) ed[e];
                            K::iq_mmvq_f32_batch(E_cnt, wp, dty, NFF, NE, E.dhh_b, E.do_b, E.st, 1);
                            for (int e = 0; e < E_cnt; ++e)
                                m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, E.st>>>(E.df, E.do_b + (size_t) e * NE, wgt[e], NE);
                            const cudaError_t ee = cudaGetLastError();
                            if (ee != cudaSuccess) { err = "extra GPU grouped GEMV: " + std::string(cudaGetErrorString(ee)); cudaSetDevice(0); return false; }
                        }
                    }
                    cudaMemcpyAsync(g_mg.relay_out + i * NE, E.df, (size_t) NE * sizeof(float), cudaMemcpyDeviceToHost, E.st);
                }
                // Phase B: GPU0's own share (experts with id % n_gpus == 0) while the extras work
                cudaSetDevice(0);
                cudaMemsetAsync(df, 0, (size_t) NE * sizeof(float), st);
                static void* pool_g = nullptr; static void* pool_u = nullptr; static void* pool_d = nullptr;
                static float *bg_g = nullptr, *bg_u = nullptr, *bg_h = nullptr, *bg_d = nullptr;
                static size_t pool_sz = 0;
                const size_t need_gu = (size_t) NFF * 4096, need_d = (size_t) NE * 4096;
                if (pool_sz < need_gu + need_gu + need_d) {
                    if (pool_g) { cudaFree(pool_g); cudaFree(pool_u); cudaFree(pool_d); }
                    pool_sz = need_gu + need_gu + need_d;
                    if (cudaMalloc(&pool_g, need_gu) != cudaSuccess || cudaMalloc(&pool_u, need_gu) != cudaSuccess ||
                        cudaMalloc(&pool_d, need_d) != cudaSuccess) { err = "expert pool"; return false; }
                    if (cudaMalloc(&bg_g, (size_t) 4 * NFF * 4) != cudaSuccess || cudaMalloc(&bg_u, (size_t) 4 * NFF * 4) != cudaSuccess ||
                        cudaMalloc(&bg_h, (size_t) 4 * NFF * 4) != cudaSuccess || cudaMalloc(&bg_d, (size_t) 4 * NE * 4) != cudaSuccess) { err = "expert batch pool"; return false; }
                }
                // Grouped on GPU0 too: fetch all assigned experts, then batched GEMVs.
                {
                    const uint8_t *eg[4] = { nullptr }, *eu[4] = { nullptr }, *ed[4] = { nullptr };
                    float wgt[4]; int gty = 0, uty = 0, dty = 0, E_cnt = 0;
                    for (int s = 0; s < r.n; ++s) {
                        if (g_mg.gpu_of(r.experts[s], g_mg.n_gpus) != 0) continue;
                        if (vexp_slots > 0) {
                            if (!g_vexp.get(m, l, r.experts[s], st, &eg[E_cnt], &gty, &eu[E_cnt], &uty, &ed[E_cnt], &dty, err)) {
                                err = "vexp expert " + std::to_string(r.experts[s]) + ": " + err; return false;
                            }
                            wgt[E_cnt] = r.weights[s]; ++E_cnt;
                        } else if (!compute_expert(s, pool_g, pool_u, pool_d, dh, dgate, dup, dhh, do_, df, st, "GPU0 expert")) return false;
                    }
                    if (E_cnt > 0) {
                        void* wp[4];
                        for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) eg[e];
                        K::iq_mmvq_f32_batch(E_cnt, wp, gty, NE, NFF, dh, bg_g, st);
                        for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) eu[e];
                        K::iq_mmvq_f32_batch(E_cnt, wp, uty, NE, NFF, dh, bg_u, st);
                        for (int e = 0; e < E_cnt; ++e)
                            m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(bg_g + (size_t) e * NFF, bg_u + (size_t) e * NFF, bg_h + (size_t) e * NFF, NFF);
                        for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) ed[e];
                        K::iq_mmvq_f32_batch(E_cnt, wp, dty, NFF, NE, bg_h, bg_d, st, 1);
                        for (int e = 0; e < E_cnt; ++e)
                            m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, bg_d + (size_t) e * NE, wgt[e], NE);
                        if (die("vexp batch")) return false;
                    }
                }
                // Phase C: wait for the extras, fold their partials into df
                for (int i = 0; i < (int) g_mg.extra.size(); ++i) {
                    cudaSetDevice(g_mg.extra[i].dev);
                    if (cudaStreamSynchronize(g_mg.extra[i].st) != cudaSuccess) {
                        err = "extra GPU wait: " + std::string(cudaGetErrorString(cudaGetLastError())); cudaSetDevice(0); return false;
                    }
                }
                cudaSetDevice(0);
                for (int i = 0; i < (int) g_mg.extra.size(); ++i) {
                    if (cudaMemcpyAsync(ds.d_part, g_mg.relay_out + i * NE, (size_t) NE * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
                        return die("partial H2D");
                    m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, ds.d_part, NE);
                }
                if (die("multigpu experts")) return false;
            } else {
                // ---- single GPU: all routed experts here (VRAM LRU when --vexperts N) ----
                cudaMemsetAsync(df, 0, (size_t) NE * sizeof(float), st);
                static void* pool_g = nullptr; static void* pool_u = nullptr; static void* pool_d = nullptr;
                static size_t pool_sz = 0;
                const size_t need_gu = (size_t) NFF * 4096, need_d = (size_t) NE * 4096;
                if (pool_sz < need_gu + need_gu + need_d) {
                    if (pool_g) { cudaFree(pool_g); cudaFree(pool_u); cudaFree(pool_d); }
                    pool_sz = need_gu + need_gu + need_d;
                    if (cudaMalloc(&pool_g, need_gu) != cudaSuccess || cudaMalloc(&pool_u, need_gu) != cudaSuccess ||
                        cudaMalloc(&pool_d, need_d) != cudaSuccess) { err = "expert pool"; return false; }
                }
                {
                    static float *bg_g = nullptr, *bg_u = nullptr, *bg_h = nullptr, *bg_d = nullptr;
                    if (!bg_g && (cudaMalloc(&bg_g, (size_t) 4 * NFF * 4) != cudaSuccess || cudaMalloc(&bg_u, (size_t) 4 * NFF * 4) != cudaSuccess ||
                                  cudaMalloc(&bg_h, (size_t) 4 * NFF * 4) != cudaSuccess || cudaMalloc(&bg_d, (size_t) 4 * NE * 4) != cudaSuccess)) { err = "expert batch pool"; return false; }
                    const uint8_t *eg[4] = { nullptr }, *eu[4] = { nullptr }, *ed[4] = { nullptr };
                    float wgt[4]; int gty = 0, uty = 0, dty = 0, E_cnt = 0;
                    for (int s = 0; s < r.n; ++s) {
                        if (vexp_slots > 0) {
                            if (!g_vexp.get(m, l, r.experts[s], st, &eg[E_cnt], &gty, &eu[E_cnt], &uty, &ed[E_cnt], &dty, err)) {
                                err = "vexp expert " + std::to_string(r.experts[s]) + ": " + err; return false;
                            }
                            wgt[E_cnt] = r.weights[s]; ++E_cnt;
                        } else if (!compute_expert(s, pool_g, pool_u, pool_d, dh, dgate, dup, dhh, do_, df, st, "expert")) return false;
                    }
                    if (E_cnt > 0) {
                        void* wp[4];
                        for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) eg[e];
                        K::iq_mmvq_f32_batch(E_cnt, wp, gty, NE, NFF, dh, bg_g, st); if (die("vexp batch gate")) return false;
                        for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) eu[e];
                        K::iq_mmvq_f32_batch(E_cnt, wp, uty, NE, NFF, dh, bg_u, st); if (die("vexp batch up")) return false;
                        for (int e = 0; e < E_cnt; ++e)
                            m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(bg_g + (size_t) e * NFF, bg_u + (size_t) e * NFF, bg_h + (size_t) e * NFF, NFF);
                        for (int e = 0; e < E_cnt; ++e) wp[e] = (void*) ed[e];
                        K::iq_mmvq_f32_batch(E_cnt, wp, dty, NFF, NE, bg_h, bg_d, st, 1); if (die("vexp batch down")) return false;
                        for (int e = 0; e < E_cnt; ++e)
                            m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, bg_d + (size_t) e * NE, wgt[e], NE);
                        if (die("vexp batch")) return false;
                    }
                }
            }
            const double t_shared0 = tprof ? now_ms() : 0.0;
            // shared expert (weight 1)
            if (!mmf(L.sg, dh, dgate, NE, NFF)) return false;
            if (!mmf(L.su, dh, dup, NE, NFF)) return false;
            m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(dgate, dup, dhh, NFF);
            if (!mmf(L.sd, dhh, do_, NFF, NE)) return false;
            m3k::k_scale_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(df, do_, 1.0f, NE);
            m3k::k_add<<<(NE + 1023) / 1024, 1024, 0, st>>>(xt, df, NE);
            if (die("moe_ffn")) return false;
        }
    return true;
}

bool m3_step_layer(Model& m, M3DecodeState& ds, int vexp_slots, int l, int P, float* xt, std::string& err) {
    if (!m3_layer_attn(m, ds, l, P, xt, err)) return false;
    return m3_layer_ffn(m, ds, vexp_slots, l, xt, err);
}


// ===========================================================================
// Chunked parallel prefill: C prompt tokens through the 60 layers in one pass.
// Dense layers run fully parallel (chunk GEMMs + tiled FA); sparse layers run
// attention per position (the decode machinery, which is position-correct) and
// the FFN grouped by expert so each expert's weights stream once per chunk.
//
// The chunk GEMMs write to rotating bounce buffers (c.bnc): on this driver,
// kernel stores to some of the long-lived scratch buffers are silently lost,
// while the bounce buffers are reliable.

bool m3_layer_ffn_chunk(Model& m, M3DecodeState& ds, int vexp_slots, int l, float* x_c, int C, std::string& err) {
    const int NE = ds.NE, NFF = ds.NFF;
    auto& SC = ds.scratch;
    auto& c = ds.chunk;
    cudaStream_t st = SC.st;
    auto die = [&](const char* w) -> bool { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); return true; } return false; };
    auto mmq_c = [&](const M3DecodeState::QW& w, const float* x, float* y, int n_in, int n_out, float*& out) -> bool {
        out = c.bnc[c.bnc_i];
        c.bnc_i = (c.bnc_i + 1) & 3;
        if (w.type == (int64_t) Q::Type::F32) {
            K::iq_mmq_f32_chunk_f32w((const float*) w.d, x, out, n_in, n_out, C, st);
        } else {
            if (K::iq_row_bytes((int) w.type, n_in) == 0) { err = "chunk GEMM: unsupported type " + std::to_string(w.type); return false; }
            K::iq_mmq_f32_chunk((int) w.type, w.d, x, out, n_in, n_out, C, st);
        }
        return !die("mmq_c");
    };
    const auto& L = ds.layers[l];
    m3k::k_rms<<<dim3((NE + 1023) / 1024, C), 1024, 0, st>>>(x_c, L.fn, c.h, NE, (float) m.geom.rms_eps);
    if (die("ffn rms")) return false;
    // router: [C][n_expert] chunk GEMM, top-4 per position on the host
    {
        float* dr = nullptr;
        if (cudaMalloc(&dr, (size_t) C * m.geom.n_expert * 4) != cudaSuccess) { err = "router scores"; return false; }
        K::iq_mmq_f32_chunk_f32w(L.router, c.h, dr, NE, (int) m.geom.n_expert, C, st);
        std::vector<float> sc((size_t) C * m.geom.n_expert), bias((size_t) m.geom.n_expert);
        cudaMemcpyAsync(sc.data(), dr, sc.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(bias.data(), L.bias, bias.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        cudaFree(dr);
        if (die("router D2H")) return false;
        struct PE { int pos; float w; };
        std::vector<std::vector<PE>> poslist(m.geom.n_expert);
        for (int i = 0; i < C; ++i) {
            mm::Route r = mm::route(sc.data() + (size_t) i * m.geom.n_expert, bias.data(),
                                    (int) m.geom.n_expert, (int) m.geom.n_expert_used, true, 2.0f);
            for (int s = 0; s < r.n; ++s)
                poslist[r.experts[s]].push_back({ i, r.weights[s] });
        }
        // each distinct expert once: gather its (normalized) rows, GEMM, scatter-add the weighted output
        for (size_t e = 0; e < poslist.size(); ++e) {
            if (poslist[e].empty()) continue;
            const int M = (int) poslist[e].size();
            float *edout = c.bnc[0], *eg = c.bnc[1], *eu = c.bnc[2], *eh = c.bnc[3];
            for (int j = 0; j < M; ++j)
                cudaMemcpyAsync(edout + (size_t) j * NE, c.h + (size_t) poslist[e][j].pos * NE, (size_t) NE * 4, cudaMemcpyDeviceToDevice, st);
            int64_t gty, uty, dty; uint64_t gb, ub, db;
            const uint8_t *gh = m.raw_slot_fast(l, (int) e, "ffn_gate_exps.weight", gty, gb, err),
                          *uh = m.raw_slot_fast(l, (int) e, "ffn_up_exps.weight", uty, ub, err),
                          *dhb = m.raw_slot_fast(l, (int) e, "ffn_down_exps.weight", dty, db, err);
            if (!gh || !uh || !dhb) { err = "expert " + std::to_string(e) + ": " + err; return false; }
            static void *sg = nullptr, *su = nullptr, *sd = nullptr;
            static size_t szg = 0, szu = 0, szd = 0;
            if (szg < gb) { if (sg) cudaFree(sg); if (cudaMalloc(&sg, gb) != cudaSuccess) { err = "expert staging"; return false; } szg = gb; }
            if (szu < ub) { if (su) cudaFree(su); if (cudaMalloc(&su, ub) != cudaSuccess) { err = "expert staging"; return false; } szu = ub; }
            if (szd < db) { if (sd) cudaFree(sd); if (cudaMalloc(&sd, db) != cudaSuccess) { err = "expert staging"; return false; } szd = db; }
            void *dg = sg, *du = su, *dd = sd;
            cudaMemcpyAsync(dg, gh, gb, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(du, uh, ub, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(dd, dhb, db, cudaMemcpyHostToDevice, st);
            if (gty == (int64_t) Q::Type::F32) K::iq_mmq_f32_chunk_f32w((const float*) dg, edout, eg, NE, NFF, M, st);
            else K::iq_mmq_f32_chunk((int) gty, dg, edout, eg, NE, NFF, M, st);
            if (uty == (int64_t) Q::Type::F32) K::iq_mmq_f32_chunk_f32w((const float*) du, edout, eu, NE, NFF, M, st);
            else K::iq_mmq_f32_chunk((int) uty, du, edout, eu, NE, NFF, M, st);
            for (int j = 0; j < M; ++j)
                m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(eg + (size_t) j * NFF, eu + (size_t) j * NFF, eh + (size_t) j * NFF, NFF);
            if (dty == (int64_t) Q::Type::F32) K::iq_mmq_f32_chunk_f32w((const float*) dd, eh, edout, NFF, NE, M, st);
            else K::iq_mmq_f32_chunk((int) dty, dd, eh, edout, NFF, NE, M, st);
            if (die("expert GEMV")) return false;
            for (int j = 0; j < M; ++j)
                m3k::k_scale_add<<<dim3((NE + 1023) / 1024, 1), 1024, 0, st>>>(x_c + (size_t) poslist[e][j].pos * NE, edout + (size_t) j * NE, poslist[e][j].w, NE);
        }
    }
    // shared expert over all C positions (bounce buffers)
    float *gate, *up, *df;
    if (!mmq_c(L.sg, c.h, nullptr, NE, NFF, gate)) return false;
    if (!mmq_c(L.su, c.h, nullptr, NE, NFF, up)) return false;
    float *hh = c.bnc[c.bnc_i]; c.bnc_i = (c.bnc_i + 1) & 3;
    for (int j = 0; j < C; ++j)
        m3k::k_swiglu_oai<<<(NFF + 1023) / 1024, 1024, 0, st>>>(gate + (size_t) j * NFF, up + (size_t) j * NFF, hh + (size_t) j * NFF, NFF);
    if (!mmq_c(L.sd, hh, nullptr, NFF, NE, df)) return false;
    m3k::k_scale_add<<<dim3((NE + 1023) / 1024, C), 1024, 0, st>>>(x_c, df, 1.0f, NE);
    if (die("shared expert")) return false;
    return true;
}


constexpr int M3_CHUNK_FA_HD = 128;

bool m3_prefill_chunk(Model& m, M3DecodeState& ds, int vexp_slots,
                      const int* tokens, int C,
                      std::vector<float>& logits, std::string& err, bool want_logits) {
    if (!ds.ready) { err = "decode state not initialised"; return false; }
    const int NE = ds.NE, H = ds.n_head, HKV = ds.HKV, HD = ds.HD, NFFD = ds.NFFD;
    const int P0 = ds.n_pos;
    if (P0 + C > ds.cap) { err = "context length exceeded"; return false; }
    if (C > ds.chunk.C) { err = "chunk larger than scratch"; return false; }
    if (vexp_slots > 0 && !g_vexp.init((size_t) vexp_slots, 32u << 20)) { err = "vexp arena: " + err; return false; }
    auto& SC = ds.scratch;
    auto& c = ds.chunk;
    cudaStream_t st = SC.st;
    auto die = [&](const char* w) -> bool { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); return true; } return false; };
    auto mmq_c = [&](const M3DecodeState::QW& w, const float* x, float* y, int n_in, int n_out, float*& out) -> bool {
        out = c.bnc[c.bnc_i];
        c.bnc_i = (c.bnc_i + 1) & 3;
        if (w.type == (int64_t) Q::Type::F32) {
            K::iq_mmq_f32_chunk_f32w((const float*) w.d, x, out, n_in, n_out, C, st);
        } else {
            if (K::iq_row_bytes((int) w.type, n_in) == 0) { err = "chunk GEMM: unsupported type " + std::to_string(w.type); return false; }
            K::iq_mmq_f32_chunk((int) w.type, w.d, x, out, n_in, n_out, C, st);
        }
        return !die("mmq_c");
    };
    // embed the C tokens
    {
        std::vector<int32_t> t32(tokens, tokens + C);
        cudaMemcpyAsync(c.rows, t32.data(), (size_t) C * sizeof(int32_t), cudaMemcpyHostToDevice, st);
        K::iq_embed_rows((int) ds.etype, ds.detab, ds.erow, c.rows, C, NE, c.x, st);
        if (die("embed")) return false;
    }
    for (int l = 0; l < ds.nL; ++l) {
        const auto& L = ds.layers[l];
        if (l < ds.dense_lead) {
            // ---- parallel dense layer ----
            m3k::k_rms<<<dim3((NE + 1023) / 1024, C), 1024, 0, st>>>(c.x, L.an, c.h, NE, (float) m.geom.rms_eps);
            float *q, *k, *v, *o, *gate, *up, *hh, *df;
            if (!mmq_c(L.wq, c.h, nullptr, NE, H * HD, q)) return false;
            if (!mmq_c(L.wk, c.h, nullptr, NE, HKV * HD, k)) return false;
            if (!mmq_c(L.wv, c.h, nullptr, NE, HKV * HD, v)) return false;
            m3k::k_rms<<<dim3(1, C * H), HD, 0, st>>>(q, L.qn, c.q2, HD, m.dims.eps);
            m3k::k_rms<<<dim3(1, C * HKV), HD, 0, st>>>(k, L.kn, c.k2, HD, m.dims.eps);
            m3k::k_rope_c<<<dim3(H + HKV, C), m.dims.n_rot / 2, 0, st>>>(c.q2, c.k2, H, HKV, HD,
                m.dims.n_rot, (float) P0, m.dims.rope_base, C);
            if (die("rope_c")) return false;
            float* lkv_k = ds.dkv_k + (size_t) l * HKV * ds.cap * HD;
            float* lkv_v = ds.dkv_v + (size_t) l * HKV * ds.cap * HD;
            m3k::k_kv_append_c<<<dim3(HKV, C), HD, 0, st>>>(lkv_k, c.k2, HKV, HD, ds.cap, P0, C);
            m3k::k_kv_append_c<<<dim3(HKV, C), HD, 0, st>>>(lkv_v, v, HKV, HD, ds.cap, P0, C);
            {
                static_assert(M3_CHUNK_FA_HD == 128, "FA head dim");
                static bool opted = false;
                const int smem = (32 * M3_CHUNK_FA_HD + 64 * M3_CHUNK_FA_HD * 2 + 32 * 64) * (int) sizeof(float);
                if (!opted) {
                    cudaFuncSetAttribute((const void*) m3k::k_fa_dense<32, 64, M3_CHUNK_FA_HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                    opted = true;
                }
                // grid.y is ceil(C/32): the last query tile may extend past C (the kernel OOB-guards)
                m3k::k_fa_dense<32, 64, M3_CHUNK_FA_HD><<<dim3(H, (C + 31) / 32), 256, smem, st>>>(c.q2, lkv_k, lkv_v, c.k2, v, c.a, H, HKV, HD, P0, C, ds.cap);
            }
            if (die("fa_dense")) return false;
            if (!mmq_c(L.wo, c.a, nullptr, H * HD, NE, o)) return false;
            m3k::k_add<<<dim3((NE + 1023) / 1024, C), 1024, 0, st>>>(c.x, o, NE);
            if (die("add1")) return false;
            m3k::k_rms<<<dim3((NE + 1023) / 1024, C), 1024, 0, st>>>(c.x, L.fn, c.h, NE, (float) m.geom.rms_eps);
            if (!mmq_c(L.dg, c.h, nullptr, NE, NFFD, gate)) return false;
            if (!mmq_c(L.du, c.h, nullptr, NE, NFFD, up)) return false;
            hh = c.bnc[c.bnc_i]; c.bnc_i = (c.bnc_i + 1) & 3;
            m3k::k_swiglu_oai<<<dim3((NFFD + 1023) / 1024, C), 1024, 0, st>>>(gate, up, hh, NFFD);
            if (!mmq_c(L.dd, hh, nullptr, NFFD, NE, df)) return false;
            m3k::k_add<<<dim3((NE + 1023) / 1024, C), 1024, 0, st>>>(c.x, df, NE);
            if (die("dense_ffn")) return false;
        } else {
            // ---- sparse layer: per-position attention + grouped chunk FFN ----
            for (int i = 0; i < C; ++i) {
                const int P = P0 + i;
                cudaMemcpyAsync(SC.dx, c.x + (size_t) i * NE, (size_t) NE * 4, cudaMemcpyDeviceToDevice, st);
                if (!m3_layer_attn(m, ds, l, P, SC.dx, err)) return false;
                cudaMemcpyAsync(c.x + (size_t) i * NE, SC.dx, (size_t) NE * 4, cudaMemcpyDeviceToDevice, st);
            }
            if (!m3_layer_ffn_chunk(m, ds, vexp_slots, l, c.x, C, err)) return false;
        }
    }
    ds.n_pos = P0 + C;
    if (!want_logits) return true;
    // ---- final norm + head for the last position ----
    {
        cudaMemcpyAsync(SC.dh, c.x + (size_t) (C - 1) * NE, (size_t) NE * 4, cudaMemcpyDeviceToDevice, st);
        m3k::k_rms<<<(NE + 1023) / 1024, 1024, 0, st>>>(SC.dh, ds.out_norm, SC.dh, NE, (float) m.geom.rms_eps);
        if (ds.head.type == (int64_t) Q::Type::F32)
            m3k::k_f32_mm<<<(m.geom.n_vocab + 127) / 128, 128, 0, st>>>((const float*) ds.head.d, SC.dh, SC.dout, NE, (int) m.geom.n_vocab);
        else
            K::iq_mmvq_f32((int) ds.head.type, ds.head.d, SC.dh, SC.dout, NE, (int) m.geom.n_vocab, st);
        logits.assign((size_t) m.geom.n_vocab, 0.0f);
        cudaMemcpyAsync(logits.data(), SC.dout, (size_t) m.geom.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        if (die("head copy")) return false;
    }
    return true;
}


bool m3_decode_step(Model& m, M3DecodeState& ds, int vexp_slots, int token_id,
                    std::vector<float>& logits, std::string& err, bool prefill_only = false) {
    if (!ds.ready) { err = "decode state not initialised"; return false; }
    const int NE = ds.NE, H = ds.n_head, HKV = ds.HKV, HD = ds.HD, NFF = ds.NFF, NFFD = ds.NFFD;
    const int P = ds.n_pos;
    if (P >= ds.cap) { err = "context length exceeded (" + std::to_string(P) + ")"; return false; }
    if (vexp_slots > 0 && !g_vexp.init((size_t) vexp_slots, 32u << 20)) { err = "vexp arena"; return false; }

    // persistent step scratch (allocated once in m3_decode_init): no per-step cudaMalloc/cudaFree, so no
    // per-step whole-device drain.  The router D2H already syncs the stream once per step, and the stream's
    // FIFO ordering keeps the reused buffers correct across steps.
    auto& SC = ds.scratch;
    cudaStream_t st = SC.st;
    auto die = [&](const char* w) -> bool { const cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string(w) + ": " + cudaGetErrorString(e); return true; } return false; };

    float *dx=SC.dx, *dh=SC.dh, *dq=SC.dq, *dk=SC.dk, *dv=SC.dv, *da=SC.da, *do_=SC.do_,
          *dgate=SC.dgate, *dup=SC.dup, *dhh=SC.dhh, *df=SC.df, *dout=SC.dout,
          *diq=SC.diq, *dik=SC.dik, *dq2=SC.dq2, *dk2=SC.dk2;
    unsigned char* dmask=SC.dmask;
    int32_t* dtok=SC.dtok;
    cudaMemcpyAsync(dtok, &token_id, sizeof(int32_t), cudaMemcpyHostToDevice, st);
    K::iq_embed_rows((int) ds.etype, ds.detab, ds.erow, dtok, 1, NE, dx, st); if (die("embed")) return false;

    auto mmf = [&](const M3DecodeState::QW& w, const float* f32a, float* y, int n_in, int n_out) -> bool {
        if (w.type == (int64_t) Q::Type::F32) {
            m3k::k_f32_mm<<<(n_out + 127) / 128, 128, 0, st>>>((const float*) w.d, f32a, y, n_in, n_out);
            return !die("mmf");
        }
        K::iq_mmvq_f32((int) w.type, w.d, f32a, y, n_in, n_out, st);
        return !die("mmf");
    };

    std::vector<float> scores((size_t) m.geom.n_expert), dh_host(NE), bias_host((size_t) m.geom.n_expert);
    const bool dtrace = getenv("M3_TRACE") != nullptr;
    const bool tprof = getenv("M3_TPROFILE") != nullptr;
    char dts_buf[64];
    auto dts = [&](const char* w, int l = -1) {
        if (l < 0) std::snprintf(dts_buf, sizeof dts_buf, "[step] %s", w);
        else std::snprintf(dts_buf, sizeof dts_buf, "[step] %s l=%d", w, l);
        std::fprintf(stderr, "%s\n", dts_buf); std::fflush(stderr);
    };
    // per-section ms accumulators (M3_TPROFILE): attn | ffn = router (incl. D2H+sync) | routed experts | shared expert
    double ms_attn = 0.0, ms_exp = 0.0, ms_router = 0.0, ms_routed = 0.0, ms_shared = 0.0;
    auto now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    int sparse_idx = 0;
    for (int l = 0; l < ds.nL; ++l) {
        if (!m3_step_layer(m, ds, vexp_slots, l, P, dx, err)) return false;
    }
    if (tprof) {
        std::fprintf(stderr, "[tprof] step P=%d attn=%.1fms ffn=%.1fms [router=%.1f routed=%.1f shared=%.1f]\n",
                     P, ms_attn, ms_exp, ms_router, ms_routed, ms_shared);
        std::fflush(stderr);
    }
    ds.n_pos = P + 1;
    if (prefill_only) {
        // a non-final prompt token: the KV is appended and no logits are needed, so skip the final norm + the
        // 200064-vocab head GEMV (the biggest per-prompt-token cost after the experts).  The router's D2H
        // already synced the stream this step, and the stream's FIFO order keeps the reused scratch correct.
        return true;
    }
    const double t_head0 = tprof ? now_ms() : 0.0;
    // ---- final norm + head ----
    m3k::k_rms<<<(NE + 1023) / 1024, 1024, 0, st>>>(dx, ds.out_norm, dh, NE, (float) m.geom.rms_eps);
    if (!mmf(ds.head, dh, dout, NE, (int) m.geom.n_vocab)) { err = "head"; return false; }
    logits.assign((size_t) m.geom.n_vocab, 0.0f);
    cudaMemcpyAsync(logits.data(), dout, (size_t) m.geom.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if (die("head copy")) return false;
    if (tprof) std::fprintf(stderr, "[tprof] head+logits=%.1fms (total step so far: see attn/ffn above)\n", now_ms() - t_head0);
    // scratch is persistent (freed once in m3_decode_free): no per-step cudaFree / cudaStreamDestroy
    return true;
}

}  // namespace

namespace strata {
namespace minimax {

int run(int argc, char** argv) {
    std::string model, tokens;
    int ctx = 8192, n_gen = 64;
    bool serve = false, use_gpu = true, use_quant = false; int n_layers = 0; int expert_cache = 0; int vexp_slots = 0; int n_gpus = 3; int expslots = 384; std::string dump_logits, dump_q, dump_x, dump_xa, dump_dv, dump_dq, dump_qn, dump_qr, dump_da, dump_kr;
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
        else if (k == "--vexperts") { if (!next(v)) return 2; vexp_slots = std::atoi(v.c_str()); }
        else if (k == "--gpus") { if (!next(v)) return 2; n_gpus = std::atoi(v.c_str()); }
        else if (k == "--expslots") { if (!next(v)) return 2; expslots = std::atoi(v.c_str()); }
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
        if (!(use_gpu ? forward_gpu(m, ids, n_layers, use_quant, expert_cache, vexp_slots, logits, err) : forward(m, ids, n_layers, logits, err))) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (use_quant && vexp_slots > 0) {
            const uint64_t total = g_vexp.hits + g_vexp.misses;
            std::fprintf(stderr, "vexp cache    %zu slots, %llu/%llu uses hit (%.1f%%)\n", g_vexp.n_slots,
                         (unsigned long long) g_vexp.hits, (unsigned long long) total,
                         total ? 100.0 * (double) g_vexp.hits / (double) total : 0.0);
        }
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
    // Load everything BEFORE announcing ready: upload the resident weights to the GPUs and stage the model in
    // pinned RAM (~112 GB, a few minutes).  The server waits up to 900 s for READY, so this is startup, not a
    // request - doing it on the first request instead would trip that request's 300 s silence watchdog.
    if (serve && use_gpu && use_quant) {
        std::string derr;
        const auto ts = std::chrono::steady_clock::now();
        if (!m3_decode_init(m, g_ds, true, ctx, n_gpus, expslots, derr)) {
            std::fprintf(stderr, "decode init failed: %s\n", derr.c_str());
            return 1;
        }
        std::fprintf(stderr, "m3 decode     weights resident + model staged in %.1f s; ready to serve\n", secs(ts));
        std::fflush(stderr);
    }
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
        if (line.rfind("GEN", 0) != 0) { std::printf("ERR expected: GEN <max> [k=v] <id,id,...>\n"); std::fflush(stdout); continue; }
        // GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N] <id,id,...>: the serve protocol rides
        // per-request sampling on the same line (serve/server.py's sampling_keys).  Unknown keys are accepted
        // and ignored (no penalty pass yet); no temperature means greedy, as always.
        std::vector<std::string> fields;
        {
            std::istringstream ls(line);
            std::string ft;
            while (std::getline(ls, ft, ' ')) if (!ft.empty()) fields.push_back(ft);
        }
        if (fields.size() < 2) continue;
        // fields[0] is the literal "GEN"; fields[1] is max_new; fields[2..] are the sampling keys and the id list
        const int max_new = std::atoi(fields[1].c_str());
        double temperature = 0.0;
        float top_p = 1.0f;
        int top_k = 0;
        float min_p = 0.0f;
        static std::mt19937 rng{std::random_device{}()};
        std::string ids_field;
        for (size_t fi = 2; fi < fields.size(); ++fi) {
            const std::string& f = fields[fi];
            const size_t eq = f.find('=');
            if (eq == std::string::npos) { ids_field = f; continue; }   // the comma-separated id list has no '='
            const std::string key = f.substr(0, eq);
            const double val = std::atof(f.c_str() + eq + 1);
            if (key == "temperature") temperature = val;
            else if (key == "top_p") top_p = (float) val;
            else if (key == "top_k") top_k = (int) val;
            else if (key == "min_p") min_p = (float) val;
            else if (key == "seed") rng.seed((unsigned) val);
        }
        if (ids_field.empty()) continue;
        std::vector<int> ids;
        size_t p = 0;
        while (p <= ids_field.size()) {
            const size_t c = ids_field.find(',', p);
            const std::string s = ids_field.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!s.empty()) ids.push_back(std::atoi(s.c_str()));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int> produced;
        if (use_gpu && use_quant) {
            // the decode engine: resident weights + a persistent KV cache, one step per token (no prefix
            // recomputation, no per-token weight re-upload).  g_ds is initialised at startup (before READY);
            // the fallback covers a serve start that skipped it.
            if (!g_ds.ready && !m3_decode_init(m, g_ds, true, ctx, n_gpus, expslots, err)) {
                std::printf("ERR decode init (try a smaller --ctx): %s\n", err.c_str()); std::fflush(stdout); continue;
            }
            M3DecodeState& ds = g_ds;
            ds.n_pos = 0;   // a new conversation per GEN
            const bool trace = getenv("M3_TRACE") != nullptr;
            const bool dbg = getenv("M3_DBG") != nullptr;   // M3_DBG=1: step-level tracing to stderr (the engine log)
            static int trace_seq = 0;
            auto tstamp = [&]() {
                const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count() % 1000000000L;
                std::fprintf(stderr, "[trace seq=%d pos=%d t=%ld]\n", trace_seq, ds.n_pos, ms); std::fflush(stderr);
            };
            std::vector<float> logits;
            bool ok = true;
            const int chunk_sz = getenv("M3_CHUNK") ? atoi(getenv("M3_CHUNK")) : 0;
            const auto prompt_t0 = std::chrono::steady_clock::now();
            if (dbg) { std::fprintf(stderr, "[dbg] GEN start: prompt=%zu ids, max_new=%d temperature=%f top_p=%f top_k=%d\n",
                                    ids.size(), max_new, temperature, (double) top_p, top_k); std::fflush(stderr); }
            for (size_t i = 0; i < ids.size(); i += (chunk_sz > 32 ? (size_t) chunk_sz : 1)) {
                const size_t natural_end = std::min(ids.size(), i + (chunk_sz > 32 ? (size_t) chunk_sz : 1));
                // the FA kernel's query tile is 32 rows: only chunk when a full tile remains
                const size_t i_end = (chunk_sz > 32 && ids.size() - i >= 32) ? natural_end : i + 1;
                const bool last_prompt = (i_end == ids.size());
                if (i_end - i >= 32) {
                    if (!m3_prefill_chunk(m, ds, vexp_slots, ids.data() + i, (int) (i_end - i), logits, err, last_prompt)) {
                        std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); ok = false; break;
                    }
                } else {
                if (trace) { tstamp(); ++trace_seq; }
                if (dbg && (i == 0 || (i + 1) % 32 == 0 || i + 1 == ids.size()))
                    { std::fprintf(stderr, "[dbg] prompt step %zu/%zu begin (P=%d)\n", i + 1, ids.size(), ds.n_pos); std::fflush(stderr); }
                // prefill: skip the final norm + 200k-vocab head for every prompt token but the last (only the
                // last needs logits for the first generated token)
                const bool last_prompt = (i + 1 == ids.size());
                if (!m3_decode_step(m, ds, vexp_slots, ids[i], logits, err, !last_prompt)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); ok = false; break; }
                if (dbg && (i == 0 || (i + 1) % 32 == 0 || i + 1 == ids.size()))
                    { std::fprintf(stderr, "[dbg] prompt step %zu/%zu done (P=%d)\n", i + 1, ids.size(), ds.n_pos); std::fflush(stderr); }
                // prompt progress (serve/server.py's PP line): the decode engine reads the prompt one token at a
                // time with no T lines until generation starts, so without this the server's silence watchdog
                // would end the engine mid-prompt on a long prompt.  Every PP line resets that timer.
                const long prompt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - prompt_t0).count();
                std::printf("PP %zu %zu %ld\n", i + 1, ids.size(), prompt_ms);
                std::fflush(stdout);
                }
                // prompt progress (serve/server.py's PP line): every PP line resets the server's watchdog
                const long prompt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - prompt_t0).count();
                std::printf("PP %zu %zu %ld\n", i_end, ids.size(), prompt_ms);
                std::fflush(stdout);
            }
            if (dbg) { std::fprintf(stderr, "[dbg] prompt done (P=%d), starting generation\n", ds.n_pos); std::fflush(stderr); }
            // the prompt's routing is free data about which experts generation will want: re-point the expert
            // cache at the hottest ones (LFU refresh) before the first generated token
            if (ok && vexp_slots > 0) {
                const auto tw0 = std::chrono::steady_clock::now();
                const size_t nloaded = g_vexp.refresh(m, nullptr, err);
                std::fprintf(stderr, "vexp refresh  %zu experts loaded in %.2f s (%zu pinned, %zu slots, %llu/%llu uses hit so far)\n",
                             nloaded, secs(tw0), g_vexp.pinned.size(), g_vexp.n_slots,
                             (unsigned long long) g_vexp.hits, (unsigned long long) (g_vexp.hits + g_vexp.misses));
                std::fflush(stderr);
            }
            if (ok) {
                for (int g = 0; g < max_new; ++g) {
                    if (dbg) { std::fprintf(stderr, "[dbg] gen step %d: pick_token begin (P=%d)\n", g, ds.n_pos); std::fflush(stderr); }
                    const int best = pick_token(logits, temperature, top_p, top_k, min_p, rng);
                    if (dbg) { std::fprintf(stderr, "[dbg] gen step %d: picked %d, emitting T line\n", g, best); std::fflush(stderr); }
                    produced.push_back(best);
                    std::printf("T %d\n", best);
                    std::fflush(stdout);
                    if (g + 1 < max_new) {
                        if (trace) { tstamp(); ++trace_seq; }
                        if (dbg) { std::fprintf(stderr, "[dbg] gen step %d: decode_step begin for token %d\n", g, best); std::fflush(stderr); }
                        if (!m3_decode_step(m, ds, vexp_slots, best, logits, err)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); break; }
                        if (dbg) { std::fprintf(stderr, "[dbg] gen step %d: decode_step done (P=%d)\n", g, ds.n_pos); std::fflush(stderr); }
                    }
                }
                if (dbg) { std::fprintf(stderr, "[dbg] generation done: %zu tokens, emitting DONE\n", produced.size()); std::fflush(stderr); }
            }
        } else {
            for (int g = 0; g < max_new; ++g) {
                std::vector<float> logits;
                if (!(use_gpu ? forward_gpu(m, ids, 0, use_quant, expert_cache, vexp_slots, logits, err) : forward(m, ids, 0, logits, err))) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); break; }
                const int best = pick_token(logits, temperature, top_p, top_k, min_p, rng);
                produced.push_back(best);
                std::printf("T %d\n", best);
                std::fflush(stdout);
                ids.push_back(best);
            }
        }
        if (use_gpu && use_quant && vexp_slots > 0) {
            const uint64_t t_use = g_vexp.hits + g_vexp.misses;
            std::fprintf(stderr, "vexp cache    GPU0 %zu slots, %zu pinned, %llu/%llu uses hit (%.1f%%)\n",
                         g_vexp.n_slots, g_vexp.pinned.size(), (unsigned long long) g_vexp.hits,
                         (unsigned long long) t_use, t_use ? 100.0 * (double) g_vexp.hits / (double) t_use : 0.0);
            std::fflush(stderr);
        }
        if (use_gpu && use_quant) {
            for (size_t i = 0; i < g_mg.extra.size(); ++i) {
                const auto& C = g_mg.extra[i].cache;
                const uint64_t t_use = C.hits + C.misses;
                if (C.n_slots == 0 || t_use == 0) continue;
                std::fprintf(stderr, "vexp cache    GPU%d %zu slots, %zu pinned, %llu/%llu uses hit (%.1f%%)\n",
                             g_mg.extra[i].dev, C.n_slots, C.pinned.size(), (unsigned long long) C.hits,
                             (unsigned long long) t_use, 100.0 * (double) C.hits / (double) t_use);
            }
            std::fflush(stderr);
        }
        std::printf("DONE %d %zu %.1f 0.0 %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", (long long) produced.size(), ids.size(),
                    secs(t0), produced.empty() ? "" : "stop", ids.size(), (long long) 0);
        std::fflush(stdout);
        if (stop.load()) break;
    }
}

}  // namespace minimax
}  // namespace strata
