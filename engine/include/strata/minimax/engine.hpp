// include/strata/minimax/engine.hpp - the engine path for MiniMax-M3.1-Flash (GGUF architecture minimax-m3).
//
// Mirrors include/strata/qwen35/engine.hpp: a `run()` that the CUDA/HIP builds of `strata` switch to at the top of
// main() when the file's architecture is minimax-m3 (see "THE FAMILY SWITCH" in src/program/generate.cpp). The model
// is a 60-layer MoE - 3 leading dense layers, 57 sparse ones (128 routed experts top-4 + 1 shared), GQA 64/4 heads
// at head_dim 128, partial RoPE over 64 of 128 dims, no attention output gate, and per-head top-16-of-128-block
// sparse attention (MSA) with a separate indexer.
//
// The forward math is the one validated by tests/minimax + tools/m3ref/m3_native (which matches the reference
// layer-by-layer and on the next token). This header only declares the entry point; the implementation is
// src/minimax/m3_engine.cu.
#pragma once

namespace strata {
namespace minimax {

/// The MiniMax-M3.1-Flash engine entry point. Same argv contract as strata::qwen35::run:
///   --serve                    serve/server.py's engine protocol (READY / GEN / T / DONE / STOP)
///   --model <shard>            the GGUF (shard 1; the rest are found via the split table)
///   --tokens <ids> [--n N]     run a prompt and print the next token(s)
///   --ctx <n>                  context size
/// Returns the process exit code.
int run(int argc, char** argv);

}  // namespace minimax
}  // namespace strata
