// tests/core/model_family_test.cpp - the model families (NEW-MODELS-PLAN.md T0) on synthetic GGUFs: no model, no GPU.
//
//   * the architecture guard accepts qwen4exp and qwen35moe and refuses anything else; the engine's guard refuses
//     qwen35moe with "not ... yet" (no engine path for Qwen3.6 before T1);
//   * MiniMax-M3.1-Flash (minimax-m3) is recognised, guarded by its own keys, read into a geometry (no GDN, no
//     gated residual, 3 dense layers first, an indexer on the rest), passed by its own layout table and still
//     refused as a shape the compiled kernels were not built for;
//   * geometry_from_gguf reads Qwen3.6-35B-A3B's real metadata values (hidden 2048, 40 + 1 MTP layers, 32 GDN V
//     heads, conv 8192, no hyper-connections, no indexer) and Flash-Next's as exactly the compiled defaults;
//   * the layout tables, on tiny files of either family: a complete file passes, a wrong shape, a missing tensor
//     and (qwen35moe) a tensor the table does not name are refused, naming the tensor.
//
// `model_family_test --real SHARD`: the same view of a real model (headers only): family, geometry, the kernels'
// verdict and the layout check over every shard.
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-model-family-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;
const uint32_t F32 = 0;

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, experts, used, ff, ff_sh, conv, state, groups, v_heads,
        inner, interval, nextn, hc, hc_lr, idx_heads, idx_key;
};
// Qwen3.6-35B-A3B (unsloth/Qwen3.6-35B-A3B-MTP-GGUF, every quant) and Qwen3.8-Flash-Next, as their files say
const Dims kQwen36 = {41, 2048, 16, 2, 256, 256, 8, 512, 512, 4, 128, 16, 32, 4096, 4, 1, 0, 0, 0, 0};
const Dims kFlashNext = {48, 2560, 24, 2, 256, 512, 10, 640, 640, 4, 128, 16, 48, 6144, 4, 0, 4, 320, 4, 128};
// tiny models of each family for the layout checks (every tensor F32, a few KB)
const Dims kTiny36 = {5, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 1, 0, 0, 0, 0};
const Dims kTinyFlash = {8, 64, 2, 1, 16, 4, 2, 32, 32, 4, 8, 2, 4, 32, 4, 0, 2, 8, 2, 8};

std::vector<Kv> metadata(const char* arch, const Dims& d, uint64_t vocab) {
    const std::string p = std::string(arch) + ".";
    std::vector<Kv> kv = {fixture::str("general.architecture", arch),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", d.heads_kv),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff),
                          fixture::u32(p + "expert_shared_feed_forward_length", d.ff_sh),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "ssm.state_size", d.state),
                          fixture::u32(p + "ssm.group_count", d.groups),
                          fixture::u32(p + "ssm.time_step_rank", d.v_heads),
                          fixture::u32(p + "ssm.inner_size", d.inner),
                          fixture::u32(p + "full_attention_interval", d.interval)};
    if (std::string(arch) == "qwen35moe") {
        kv.push_back(fixture::u32(p + "nextn_predict_layers", d.nextn));
    } else {
        kv.push_back(fixture::u32(p + "hyper_connection.count", d.hc));
        kv.push_back(fixture::u32(p + "hyper_connection.low_rank", d.hc_lr));
        kv.push_back(fixture::u32(p + "attention.indexer.head_count", d.idx_heads));
        kv.push_back(fixture::u32(p + "attention.indexer.key_length", d.idx_key));
    }
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// MiniMax-M3.1-Flash's metadata as its file says it (AtomicChat/MiniMax-M3.1-Flash-GGUF, AD-* shards): 60 layers,
/// the first 3 dense, 6144 wide, 64 heads over 4 KV heads of 128, 128 experts with 4 + 1 shared of 3072, and an
/// indexer that picks 16 blocks of 128 keys.  No tensors: the family and the guards need the header only.
struct M3Dims {
    uint64_t layers, dense, embd, ffn_dense, heads, heads_kv, head, rot, experts, used, e_ff, idx_heads, idx_key,
        idx_topk, idx_block, idx_local;
};
// the real model, and a five-layer stand-in whose tensors are a few KB (head_dim 16 keeps the same shape rules)
const M3Dims kM3Real = {60, 3, 6144, 12288, 64, 4, 128, 64, 128, 4, 3072, 4, 128, 16, 128, 1};
const M3Dims kM3Tiny = {5, 2, 64, 48, 4, 2, 16, 8, 4, 2, 32, 2, 8, 4, 4, 1};

std::vector<Kv> minimax_metadata(const M3Dims& d, uint64_t vocab = 8) {
    std::vector<Kv> kv = {fixture::str("general.architecture", "minimax-m3"),
                          fixture::u32("minimax-m3.block_count", d.layers),
                          fixture::u32("minimax-m3.leading_dense_block_count", d.dense),
                          fixture::u32("minimax-m3.embedding_length", d.embd),
                          fixture::u32("minimax-m3.feed_forward_length", d.ffn_dense),
                          fixture::u32("minimax-m3.attention.head_count", d.heads),
                          fixture::u32("minimax-m3.attention.head_count_kv", d.heads_kv),
                          fixture::u32("minimax-m3.attention.key_length", d.head),
                          fixture::u32("minimax-m3.attention.value_length", d.head),
                          fixture::u32("minimax-m3.rope.dimension_count", d.rot),
                          fixture::u32("minimax-m3.expert_count", d.experts),
                          fixture::u32("minimax-m3.expert_used_count", d.used),
                          fixture::u32("minimax-m3.expert_shared_count", 1),
                          fixture::u32("minimax-m3.expert_feed_forward_length", d.e_ff),
                          // the two keys that decide the MATH, not the shapes - the real AD shard says 2 and 1
                          fixture::u32("minimax-m3.expert_gating_func", 2),   // 2 = sigmoid gating
                          fixture::u32("minimax-m3.expert_weights_norm", 1),  // and the selected weights are normalised
                          fixture::u32("minimax-m3.attention.indexer.head_count", d.idx_heads),
                          fixture::u32("minimax-m3.attention.indexer.key_length", d.idx_key),
                          fixture::u32("minimax-m3.attention.indexer.top_k", d.idx_topk),
                          fixture::u32("minimax-m3.attention.indexer.block_size", d.idx_block),
                          fixture::u32("minimax-m3.attention.indexer.local_blocks", d.idx_local)};
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}
std::vector<Kv> minimax_metadata() { return minimax_metadata(kM3Real); }

/// The tensors the layout table of `g` names, as F32 fixtures.
std::vector<Tensor> tensors_of(const strata::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        std::vector<uint64_t> s(shape.begin(), shape.end());
        ts.push_back({name, s, F32, seed++});
    };
    for (const auto& w : strata::core::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : strata::core::layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

bool geometry_of(const fs::path& p, strata::core::ModelGeometry& g, int64_t& k, std::string& err) {
    const strata::GgufFile f(p.string());
    return strata::core::geometry_from_gguf(f, g, k, err);
}

std::string layout_error(const fs::path& dir, const char* arch, const Dims& d, uint64_t vocab,
                         void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / "meta.gguf";
    fixture::write(meta, metadata(arch, d, vocab), {});
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    if (std::string(arch) == "qwen35moe") {
        if (!geometry_of(meta, g, k, err)) return "geometry: " + err;
    } else {
        // the qwen4exp guard pins Flash-Next's own block count and widths, so the tiny model's geometry is set here
        g.n_layers = (int64_t) d.block_count;
        g.n_embd = (int64_t) d.embd;
        g.n_head = (int64_t) d.heads;
        g.n_head_kv = (int64_t) d.heads_kv;
        g.head_dim = (int64_t) d.key_len;
        g.n_expert = (int64_t) d.experts;
        g.n_ff = (int64_t) d.ff;
        g.n_ff_shexp = (int64_t) d.ff_sh;
        g.ssm_d_conv = (int64_t) d.conv;
        g.ssm_state_size = (int64_t) d.state;
        g.ssm_k_heads = (int64_t) d.groups;
        g.ssm_v_heads = (int64_t) d.v_heads;
        g.ssm_value_dim = (int64_t) d.inner;
        g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
        g.hc = (int64_t) d.hc;
        g.hc_lr = (int64_t) d.hc_lr;
        g.idx_q_heads = (int64_t) d.idx_heads;
        g.idx_key_dim = (int64_t) d.idx_key;
        g.n_vocab = (int64_t) vocab;
    }
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / (std::string(arch) + "-tiny.gguf");
    fixture::write(model, metadata(arch, d, vocab), ts);
    const strata::GgufModel m({model.string()});
    size_t n = 0;
    if (!strata::core::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}

/// The tiny M3 model's layout: its geometry from its own metadata, its tensors from the table, and the verdict.
std::string minimax_layout(const fs::path& dir, const char* tag, void (*mutate)(std::vector<Tensor>&)) {
    const fs::path meta = dir / (std::string(tag) + "-meta.gguf");
    fixture::write(meta, minimax_metadata(kM3Tiny, 8), {});
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    if (!geometry_of(meta, g, k, err)) return "geometry: " + err;
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / (std::string(tag) + "-tiny.gguf");
    fixture::write(model, minimax_metadata(kM3Tiny, 8), ts);
    const strata::GgufModel m({model.string()});
    size_t n = 0;
    if (!strata::core::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}

int real_model(const std::string& any) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(any);
        const std::string arch = strata::check_architecture(m.meta());
        std::printf("  %zu shard(s); architecture guard: %s; engine guard: %s\n", m.size(),
                    arch.empty() ? "ok" : arch.c_str(),
                    strata::check_engine_architecture(m.meta()).empty() ? "runs"
                                                                         : strata::check_engine_architecture(m.meta()).c_str());
        if (!arch.empty()) return 1;
        strata::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        if (!strata::core::geometry_from_gguf(m.meta(), g, k, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("  family %s: layers %lld (+%lld MTP), hidden %lld, attention %lld/%lld x %lld every %lld, GDN K %lld"
                    " V %lld x %lld conv %lld (%lld), MoE %lld experts top-%lld FF %lld shared %lld, hc %lld, indexer"
                    " %lld, vocab %lld\n",
                    strata::model_family_title(g.family), (long long) g.n_layers, (long long) g.n_mtp_layers,
                    (long long) g.n_embd, (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim,
                    (long long) g.qsa_interval, (long long) g.ssm_k_heads, (long long) g.ssm_v_heads,
                    (long long) g.ssm_state_size, (long long) g.ssm_conv_channels, (long long) g.ssm_d_conv,
                    (long long) g.n_expert, (long long) k, (long long) g.n_ff, (long long) g.n_ff_shexp,
                    (long long) g.hc, (long long) g.idx_q_heads, (long long) g.n_vocab);
        const std::string mm = strata::core::compiled_geometry_mismatch(g);
        std::printf("  compiled kernels: %s\n", mm.empty() ? "match" : mm.c_str());
        size_t n = 0;
        const bool ok = strata::core::check_gguf_layout(m, g, err, &n);
        std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("model_family_test\n");
    TempDir d;
    {
        // the guards
        const fs::path q36 = d.path / "q36.gguf", fn = d.path / "fn.gguf", other = d.path / "other.gguf";
        fixture::write(q36, metadata("qwen35moe", kQwen36, 8), {});
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 8), {});
        fixture::write(other, {fixture::str("general.architecture", "llama")}, {});
        const strata::GgufFile a(q36.string()), b(fn.string()), c(other.string());
        check(strata::model_family(a) == strata::ModelFamily::Qwen35Moe &&
                  strata::model_family(b) == strata::ModelFamily::Qwen4Exp &&
                  strata::model_family(c) == strata::ModelFamily::Unknown,
              "model_family: qwen35moe, qwen4exp, unknown");
        check(strata::check_architecture(a).empty(), "the architecture guard accepts qwen35moe");
        check(strata::check_architecture(b).empty(), "the architecture guard accepts qwen4exp");
        check(strata::check_architecture(c).find("'llama'") != std::string::npos, "and refuses llama, naming it");
        const std::string e = strata::check_engine_architecture(a);
        check(e.find("Qwen3.6-35B-A3B") != std::string::npos && e.find("does not run it yet") != std::string::npos,
              "the engine guard refuses Qwen3.6: recognised, not supported yet");
        check(strata::check_engine_architecture(b).empty(), "the engine guard runs qwen4exp");
        std::vector<Kv> kv = metadata("qwen35moe", kQwen36, 8);
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "qwen35moe.ssm.time_step_rank") { kv.erase(it); break; }
        const fs::path miss = d.path / "miss.gguf";
        fixture::write(miss, kv, {});
        check(strata::check_architecture(strata::GgufFile(miss.string())).find("ssm.time_step_rank") != std::string::npos,
              "a qwen35moe file without ssm.time_step_rank is refused, naming the key");
    }
    {
        // MiniMax-M3.1-Flash: recognised, guarded by its own keys, and refused as a family with no engine yet
        const fs::path mm = d.path / "minimax.gguf";
        fixture::write(mm, minimax_metadata(), {});
        const strata::GgufFile m(mm.string());
        check(strata::model_family(m) == strata::ModelFamily::MiniMaxM3, "model_family: minimax-m3 is MiniMaxM3");
        check(strata::check_architecture(m).empty(), "the architecture guard accepts a MiniMax-M3.1-Flash file");
        const std::string e = strata::check_engine_architecture(m);
        check(e.find("MiniMax-M3.1-Flash") != std::string::npos && e.find("no engine for yet") != std::string::npos,
              "the engine guard refuses it as recognised without an engine: " + e);
        check(e.find("Intel Arc") == std::string::npos && e.find("CUDA") == std::string::npos,
              "and does not send the person to another build that cannot run it either");
        std::vector<Kv> kv = minimax_metadata();
        for (auto it = kv.begin(); it != kv.end(); ++it)
            if (it->key == "minimax-m3.attention.indexer.top_k") { kv.erase(it); break; }
        const fs::path noidx = d.path / "minimax-noidx.gguf";
        fixture::write(noidx, kv, {});
        check(strata::check_architecture(strata::GgufFile(noidx.string())).find("indexer.top_k") != std::string::npos,
              "a minimax-m3 file without attention.indexer.top_k is refused, naming the key");
        kv = minimax_metadata();
        for (auto& e2 : kv)
            if (e2.key == "minimax-m3.expert_used_count") e2 = fixture::u32(e2.key, 200);
        const fs::path wide = d.path / "minimax-wide.gguf";
        fixture::write(wide, kv, {});
        check(strata::check_architecture(strata::GgufFile(wide.string())).find("expert_used_count") != std::string::npos,
              "more experts used than the file has experts is refused, naming the key");
        // gating and normalisation change what the router computes, so a file that asks for something else is
        // refused by name rather than routed with sigmoid anyway
        kv = minimax_metadata();
        for (auto& e2 : kv)
            if (e2.key == "minimax-m3.expert_gating_func") e2 = fixture::u32(e2.key, 1);   // 1 = softmax
        fixture::write(d.path / "minimax-softmax.gguf", kv, {});
        check(strata::check_architecture(strata::GgufFile((d.path / "minimax-softmax.gguf").string()))
                  .find("expert_gating_func") != std::string::npos,
              "a MiniMax file that asks for softmax gating is refused, naming the key");
        decltype(kv) nonorm;
        for (const auto& e2 : kv)
            if (e2.key != "minimax-m3.expert_weights_norm") nonorm.push_back(e2);
        fixture::write(d.path / "minimax-nonorm.gguf", nonorm, {});
        check(strata::check_architecture(strata::GgufFile((d.path / "minimax-nonorm.gguf").string()))
                  .find("expert_weights_norm") != std::string::npos,
              "and one that drops expert_weights_norm is refused too, because the reference requires it");
    }
    {
        // MiniMax-M3.1-Flash's geometry and layout table (the numbers its AD files carry)
        const fs::path p = d.path / "minimax-geo.gguf";
        fixture::write(p, minimax_metadata(kM3Real, 8), {});
        strata::core::ModelGeometry g;
        int64_t top_k = 0;
        std::string err;
        const bool ok = geometry_of(p, g, top_k, err);
        check(ok, "M3 geometry reads" + (ok ? std::string() : ": " + err));
        check(g.family == strata::ModelFamily::MiniMaxM3 && g.n_layers == 60 && g.n_embd == 6144 && top_k == 4,
              "M3: 60 layers, hidden 6144, 4 of 128 experts per token");
        check(g.n_head == 64 && g.n_head_kv == 4 && g.head_dim == 128 && g.n_rot == 64,
              "M3 attention: 64 Q / 4 KV x 128, 64 rotated dims");
        check(g.n_ff == 3072 && g.n_ff_shexp == 3072 && g.n_ff_dense == 12288,
              "M3 MoE: expert FF 3072, the shared expert the same width (the file names no key for it), dense FF 12288");
        check(g.idx_q_heads == 4 && g.idx_key_dim == 128 && g.idx_top_k == 16 && g.idx_block == 128 &&
                  g.idx_local_blocks == 1,
              "M3 indexer: 4 heads of 128, 16 blocks of 128 keys, 1 local block");
        check(g.qsa_interval == 1 && g.n_gdn_layers() == 0 && g.n_qsa_layers() == 60 && !g.has_hyper_connections(),
              "M3: no GDN and no gated residual; every layer attends over the whole past");
        check(g.n_dense_lead == 3 && !g.has_layer_indexer(0) && !g.has_layer_indexer(2) && g.has_layer_indexer(3) &&
                  g.has_dense_ffn(2) && !g.has_dense_ffn(3),
              "M3: layers 0-2 are dense FFN with full attention, layer 3 on is sparse MoE with an indexer");
        auto names = [&](int64_t l) {
            std::string s;
            for (const auto& w : strata::core::layer_tensors(g, l)) s += w.name + ";";
            return s;
        };
        check(names(0).find("ffn_gate.weight") != std::string::npos &&
                  names(0).find("_exps.") == std::string::npos && names(0).find("indexer.") == std::string::npos,
              "the table gives a dense M3 layer a plain MLP and no indexer");
        // exact names, not substrings: a renamed or suffixed tensor would still satisfy a find()
        // exact entries, not loose substrings: `;name;` only matches a whole row of the table, so a renamed or
        // suffixed tensor can no longer satisfy the check by accident
        auto holds = [&](int64_t l, const char* n) {
            return (";" + names(l)).find(std::string(";") + n + ";") != std::string::npos;
        };
        check(holds(3, "ffn_gate_inp.weight") && holds(3, "ffn_gate_exps.weight") && holds(3, "exp_probs_b.bias") &&
                  holds(3, "ffn_gate_shexp.weight") && holds(3, "ffn_down_shexp.weight") &&
                  holds(3, "indexer.q_proj.weight") && holds(3, "indexer.k_norm.weight") &&
                  !holds(3, "ffn_gate.weight") && !holds(3, "attn_q_gate.weight"),
              "and a sparse one the router, the sigmoid bias, routed and shared experts and the indexer, no dense MLP");
        check(!strata::core::compiled_geometry_mismatch(g).empty() &&
                  strata::core::compiled_geometry_mismatch(g).find("MiniMax-M3.1-Flash") != std::string::npos,
              "the kernels still say they are built for another shape: " + strata::core::compiled_geometry_mismatch(g));
        check(minimax_layout(d.path, "m3ok", nullptr).empty(), "a complete M3 file passes its layout table");
        check(minimax_layout(d.path, "m3extra", [](std::vector<Tensor>& ts) {
                  ts.push_back({"blk.3.attn_gate.weight", {64, 64}, F32, 200});
              }).find("attn_gate") != std::string::npos,
              "an M3 tensor the table does not name is refused, naming it");
        check(minimax_layout(d.path, "m3qwen", [](std::vector<Tensor>& ts) {
                  for (auto& x : ts)
                      if (x.name == "blk.3.attn_q.weight") x.shape = {64, 2 * 4 * 16};   // Qwen's gated q: not M3's
              }).find("attn_q") != std::string::npos,
              "M3's attn_q with Qwen's output-gate width is refused, naming the tensor");
    }
    {
        // the geometry of the real models, from their metadata values
        const fs::path q36 = d.path / "q36g.gguf", fn = d.path / "fng.gguf";
        fixture::write(q36, metadata("qwen35moe", kQwen36, 248320), {});
        fixture::write(fn, metadata("qwen4exp", kFlashNext, 248320), {});
        strata::core::ModelGeometry g;
        int64_t k = 0;
        std::string err;
        const bool ok = geometry_of(q36, g, k, err);
        check(ok, "Qwen3.6 geometry reads" + (ok ? std::string() : ": " + err));
        check(g.family == strata::ModelFamily::Qwen35Moe && g.n_layers == 40 && g.n_mtp_layers == 1 && g.n_embd == 2048,
              "Qwen3.6: 40 trunk layers + 1 MTP block, hidden 2048");
        check(g.n_qsa_layers() == 10 && g.n_gdn_layers() == 30 && strata::core::layer_kind(g, 3) ==
                  strata::core::LayerKind::Attention && strata::core::layer_kind(g, 40) == strata::core::LayerKind::Mtp,
              "Qwen3.6: 10 attention / 30 GDN layers, layer 40 is the MTP block");
        check(g.ssm_k_heads == 16 && g.ssm_v_heads == 32 && g.ssm_conv_channels == 8192 && g.ssm_value_dim == 4096,
              "Qwen3.6 GDN: 16 K groups, 32 V heads, conv 8192, inner 4096");
        check(g.n_head == 16 && g.n_head_kv == 2 && g.head_dim == 256 && !g.has_indexer() && !g.has_hyper_connections(),
              "Qwen3.6 attention: 16 Q / 2 KV x 256, no indexer, no hyper-connections");
        check(g.n_expert == 256 && k == 8 && g.n_ff == 512 && g.n_ff_shexp == 512 && g.n_vocab == 248320,
              "Qwen3.6 MoE: 256 experts top-8, FF 512, shared 512; vocab 248320");
        check(!strata::core::compiled_geometry_mismatch(g).empty(), "Qwen3.6 does not match the compiled kernels");
        strata::core::ModelGeometry f;
        const bool okf = geometry_of(fn, f, k, err);
        const strata::core::ModelGeometry c;
        check(okf && strata::core::compiled_geometry_mismatch(f).empty() && k == 10 && f.n_expert == c.n_expert &&
                  f.ssm_conv_channels == c.ssm_conv_channels && f.n_vocab == c.n_vocab && f.hc_dim() == c.hc_dim(),
              "Flash-Next's metadata gives exactly the compiled geometry");
        Dims coder = kFlashNext;
        coder.experts = 256;
        const fs::path cp = d.path / "coder.gguf";
        fixture::write(cp, metadata("qwen4exp", coder, 8), {});
        check(geometry_of(cp, f, k, err) && f.n_expert == 256 && strata::core::compiled_geometry_mismatch(f).empty(),
              "the Coder's 256 experts are a run-time size, not a kernel mismatch");
        Dims wrong = kQwen36;
        wrong.inner = 4000;
        const fs::path wp = d.path / "wrong.gguf";
        fixture::write(wp, metadata("qwen35moe", wrong, 8), {});
        check(!geometry_of(wp, f, k, err) && err.find("ssm.inner_size") != std::string::npos,
              "an inconsistent ssm.inner_size is refused");
    }
    {
        // the layout tables
        check(layout_error(d.path, "qwen35moe", kTiny36, 16, nullptr).empty(),
              "a complete qwen35moe file passes its layout (incl. the nextn MTP block)");
        const std::string ef = layout_error(d.path, "qwen4exp", kTinyFlash, 16, nullptr);
        check(ef.empty(), "a complete qwen4exp file passes its layout (hyper-connections, indexer)" +
                              (ef.empty() ? std::string() : ": " + ef));
        std::string e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.1.ssm_conv1d.weight") t.shape[1] += 1;
        });
        check(e.find("blk.1.ssm_conv1d.weight") != std::string::npos && e.find("wants") != std::string::npos,
              "a wrong shape is refused, naming the tensor and both shapes");
        e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.4.nextn.eh_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.4.nextn.eh_proj.weight", "a missing MTP tensor is refused");
        e = layout_error(d.path, "qwen35moe", kTiny36, 16, [](std::vector<Tensor>& ts) {
            ts.push_back({"blk.0.hc_attn_norm.weight", {64}, F32, 99});
        });
        check(e.find("blk.0.hc_attn_norm.weight") != std::string::npos && e.find("not in the") != std::string::npos,
              "a tensor the qwen35moe table does not name is refused");
        e = layout_error(d.path, "qwen4exp", kTinyFlash, 16, [](std::vector<Tensor>& ts) {
            for (auto it = ts.begin(); it != ts.end(); ++it)
                if (it->name == "blk.3.indexer.k_proj.weight") { ts.erase(it); break; }
        });
        check(e == "missing blk.3.indexer.k_proj.weight", "Flash-Next: a missing indexer tensor is refused (" + e + ")");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
