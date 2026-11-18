// tools/m3ref/m3run.cpp - run one model once, and print the numbers a kernel plan needs.
//
// WHY THIS EXISTS. Strata has no MiniMax-M3.1-Flash kernels yet - the engine says so in the message it refuses
// with - but "does this file run on this hardware, at what speed, holding how much memory" is answerable today:
// the vendored llama.cpp fork in third_party/strata-llama implements the architecture, and Strata already builds
// its ggml with CUDA. This links that libllama, loads one GGUF (shards included), and prints load time, prefill
// and decode tokens per second, peak RSS and the greedy completion. The completion is the reference a future
// Strata CUDA path has to match token for token, so temperature is 0 unless someone asks otherwise: a reference
// has to be the model, not a sampler.
//
// Build it next to libllama, not inside Strata's build (Strata adds only ggml/, not src/):
//
//   cmake -S third_party/strata-llama -B build-m3llm -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
//         -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_COMMON=OFF
//   cmake --build build-m3llm --target llama -j12
//   g++ -O2 -std=c++17 tools/m3ref/m3run.cpp -o build-m3llm/m3run \
//       -I third_party/strata-llama/include -I third_party/strata-llama/ggml/include \
//       build-m3llm/src/libllama.a build-m3llm/ggml/src/libggml.a build-m3llm/ggml/src/libggml-base.a \
//       build-m3llm/ggml/src/ggml-cuda/libggml-cuda.a -lcudart -lcublas -lcublasLt -ldl -lpthread
//
// Example, with the routed experts kept in RAM and the attention on the cards (the shape this machine is large
// enough for and the small cards are not):
//
//   build-m3llm/m3run --model <...-00001-of-00003.gguf> -ngl 99 -c 2048 -n 64 --experts-cpu --prompt "..."
//
// Exit status is 0 only if the model loaded and at least one token came out.

#include "ggml-backend.h"
#include "llama.h"

#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model, prompt = "Hello", override_pattern, override_target;
    int32_t n_gpu_layers = 0, n_ctx = 512, n_batch = 512, n_threads = 0, n_tokens = 32, seed = 1234;
    bool experts_cpu = false, no_mmap = false;
};

std::string slurp(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
    std::string s;
    char buf[4096];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, r);
    std::fclose(f);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

const char* need(int& i, int argc, char** argv) {
    if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", argv[i]); std::exit(2); }
    return argv[++i];
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") a.model = need(i, argc, argv);
        else if (k == "--prompt") a.prompt = need(i, argc, argv);
        else if (k == "--prompt-file") a.prompt = slurp(need(i, argc, argv));
        else if (k == "--experts-cpu") a.experts_cpu = true;
        else if (k == "--override") { a.override_pattern = need(i, argc, argv); a.override_target = need(i, argc, argv); }
        else if (k == "-ngl" || k == "--n-gpu-layers") a.n_gpu_layers = std::atoi(need(i, argc, argv));
        else if (k == "-c" || k == "--ctx-size") a.n_ctx = std::atoi(need(i, argc, argv));
        else if (k == "-b" || k == "--batch-size") a.n_batch = std::atoi(need(i, argc, argv));
        else if (k == "-n" || k == "--tokens") a.n_tokens = std::atoi(need(i, argc, argv));
        else if (k == "-t" || k == "--threads") a.n_threads = std::atoi(need(i, argc, argv));
        else if (k == "--seed") a.seed = std::atoi(need(i, argc, argv));
        else if (k == "--no-mmap") a.no_mmap = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", k.c_str()); std::exit(2); }
    }
    if (a.model.empty()) { std::fprintf(stderr, "--model is required\n"); std::exit(2); }
    return a;
}

long peak_rss_mib() {
    struct rusage ru;
    ::getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;   // kilobytes on Linux
}

double since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.n_gpu_layers;
    mp.load_mode = a.no_mmap ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;

    // Keep the routed experts in RAM: they are 92% of this file's bytes and 57 x 4 slots per token, which no card
    // here can hold. NOTE the pattern is a bare tensor-name regex - the `=CPU` suffix is llama.cpp's COMMAND-LINE
    // syntax, not the API's, and passing it here makes the regex match nothing, which sends all 104 GiB to the card.
    std::vector<llama_model_tensor_buft_override> ov;
    if (a.experts_cpu) {
        ov.push_back({"\\.ffn_.*_exps\\.", ggml_backend_cpu_buffer_type()});
        ov.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ov.data();
    } else if (!a.override_pattern.empty()) {
        ov.push_back({a.override_pattern.c_str(),   // a bare regex over tensor names
                      a.override_target == "CPU" ? ggml_backend_cpu_buffer_type() : nullptr});
        ov.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ov.data();
    }

    const auto t0 = std::chrono::steady_clock::now();
    llama_model* model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { std::fprintf(stderr, "load failed: %s\n", a.model.c_str()); return 1; }
    const double load_s = since(t0);

    std::printf("model              %.2f B parameters, %.2f GiB of weights\n", llama_model_n_params(model) / 1e9,
                llama_model_size(model) / 1073741824.0);
    std::printf("load               %.1f s, peak RSS %ld MiB\n", load_s, peak_rss_mib());

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = a.n_ctx;
    cp.n_batch = a.n_batch;
    cp.n_ubatch = a.n_batch;
    if (a.n_threads > 0) {
        cp.n_threads = a.n_threads;
        cp.n_threads_batch = a.n_threads;
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }
    std::printf("ctx                %u tokens, n_batch %d, n_gpu_layers %d%s\n", llama_n_ctx(ctx), a.n_batch,
                a.n_gpu_layers, a.experts_cpu ? ", routed experts in RAM" : "");

    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(2 * (a.prompt.size() + 8));
    int32_t n_tok = llama_tokenize(vocab, a.prompt.c_str(), (int32_t) a.prompt.size(), tokens.data(),
                                   (int32_t) tokens.size(), /*add_special*/ true, /*parse_special*/ true);
    if (n_tok <= 0) { std::fprintf(stderr, "tokenization failed (%d)\n", n_tok); return 1; }
    tokens.resize(n_tok);

    const auto tp = std::chrono::steady_clock::now();
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok)) != 0) {
        std::fprintf(stderr, "prefill failed\n");
        return 1;
    }
    const double pf_s = since(tp);
    std::printf("prefill            %d tokens in %.3f s = %.2f tok/s\n", n_tok, pf_s, n_tok / (pf_s > 0 ? pf_s : 1e-9));

    // greedy: the completion is a reference, and a reference has to be the model rather than a sampler
    llama_sampler* smpl = llama_sampler_init_greedy();

    std::string out;
    int made = 0;
    const auto td = std::chrono::steady_clock::now();
    for (int i = 0; i < a.n_tokens; ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);
        char piece[256] = {0};
        const int32_t n = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, 0, false);
        if (n > 0) out.append(piece, n);
        ++made;
        if (llama_vocab_is_eog(vocab, id)) break;
        tokens.assign(1, id);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), 1)) != 0) {
            std::fprintf(stderr, "decode failed at token %d\n", i);
            break;
        }
    }
    const double dec_s = since(td);
    std::printf("decode             %d tokens in %.3f s = %.2f tok/s\n", made, dec_s,
                made / (dec_s > 0 ? dec_s : 1e-9));
    std::printf("peak RSS           %ld MiB\n", peak_rss_mib());
    std::printf("--- completion (greedy) ---\n%s\n", out.c_str());

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return made > 0 ? 0 : 1;
}
