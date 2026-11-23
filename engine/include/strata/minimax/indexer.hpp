// include/strata/minimax/indexer.hpp - MiniMax-M3.1-Flash's sparse attention: which key blocks a query reads.
//
// Header-only, dependency-free and GPU-free: the parity test (tests/minimax/indexer_test.cpp) runs it against a
// double reference anywhere, and the CUDA path takes its numbers from the same functions.
//
// M3 attends over the whole past on every layer but the 3 leading dense ones, and it does that sparsely: an indexer
// scores each block of keys and the query reads the K best blocks.  The formulas follow the model's own release
// implementation (llama.cpp at the pin in third_party/strata-llama, `models/minimax-m3.cpp`):
//
//   score      = dot(indexer_query, indexer_key)          - the raw dot, NO 1/sqrt(dim): only the order matters
//   mask       = -INFINITY on a position with no key      (holes, padding, positions not written yet)
//   block      = the max of the masked scores in the block (ggml_pool_2d with POOL_MAX, block = 128 positions)
//   local force= the blocks holding the query's own position get +1e30, `attention.indexer.local_blocks` of them,
//                so the recent window is always read no matter what the scores say
//   selection  = the K highest block scores; equal scores take the lower block index
//
// The scores run in F32 (the graph sets GGML_PREC_F32 on that matmul), and llama.cpp picks the blocks with
// `ggml_top_k`, whose comparator looks at the score alone under an unstable `std::partial_sort` - an exact tie may
// come out in any order.  Strata pins the lower index, so one token selects the same blocks on every backend.
//
// M3's own numbers, from its file: 4 indexer heads of 128, blocks of 128 positions, 16 blocks selected, 1 local
// block (16 x 128 = 2048 positions of attention budget per head - the same budget Flash-Next's QSA spends).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strata::minimax {

/// The most blocks one query may select.  M3 selects 16; the ceiling keeps the selection a fixed-size value.
constexpr int kMaxSelectedBlocks = 32;

/// The value that forces a block in, as the reference writes it.  It is not a score: any real score is far smaller.
constexpr float kLocalForceBias = 1e30f;

/// The most blocks one selection scans.  A 248 GB machine's cache at 128 positions per block is far below this; a
/// wider cache is refused here rather than overflowed, because this is the reference the kernels are checked against.
constexpr int kMaxBlocksScanned = 4096;

/// The blocks a head may read, in score order.
struct BlockSel {
    int blocks[kMaxSelectedBlocks];
    int n;   ///< filled entries; 0 means the call was refused (see `select_blocks`)
};

/// The raw dot of one indexer query and one indexer key, both `dim` wide, accumulated in float - the F32 the graph
/// asks for.  No scaling: the model only ever compares these.
inline float dot(const float* q, const float* k, int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) acc += q[i] * k[i];
    return acc;
}

/// One head's block scores.  `keys` holds `n_pos` keys of `dim` floats each, `mask` is 0.0 on a written position and
/// -INFINITY on one that is not (a hole, padding, or a tail the block size rounds up over).  The block that holds
/// query position `pos` and the `local` blocks before it are forced in.  `out` is `ceil(n_pos / blk)` wide; a block
/// with no written position at all comes out -INFINITY, so it can never be read.
inline void block_scores(const float* q, const float* keys, const float* mask, int n_pos, int dim, int blk,
                         long long pos, int local, float* out) {
    if (!q || !keys || !out || n_pos < 0 || dim <= 0 || blk <= 0) return;
    const int n_blk = (n_pos + blk - 1) / blk;
    for (int b = 0; b < n_blk; ++b) {
        float best = -INFINITY;
        const int first = b * blk, last = first + blk < n_pos ? first + blk : n_pos;
        for (int p = first; p < last; ++p) {
            const float m = mask ? mask[p] : 0.0f;
            if (m == -INFINITY) continue;   // no key here: it cannot carry a block into the selection
            const float s = dot(q, keys + (size_t)p * dim, dim) + m;
            if (s > best) best = s;
        }
        out[b] = best;
    }
    const long long own = static_cast<long long>(pos) / blk;
    for (int l = 0; l < local && own - l >= 0; ++l) {
        const long long b = own - l;
        if (b < n_blk) out[b] += kLocalForceBias;
    }
}

/// The `k` best blocks of one head, highest score first; equal scores take the lower block index.  A block scored
/// -INFINITY is returned only when there is nothing else, which is what a query before any key sees.
/// Note this returns `min(k, n_blk)` rows, where ggml_top_k asserts ne[0] >= k and always returns K rows over the
/// whole cache; a caller that assumes K will read unselected entries on a short sequence.
inline BlockSel select_blocks(const float* scores, int n_blk, int k) {
    BlockSel s;
    s.n = 0;
    if (!scores || n_blk <= 0 || k <= 0 || k > kMaxSelectedBlocks) return s;
    bool taken[kMaxBlocksScanned];
    if (n_blk > kMaxBlocksScanned) return s;   // a wider cache is scored elsewhere, not overflowed here
    for (int b = 0; b < n_blk; ++b) taken[b] = false;
    const int want = k < n_blk ? k : n_blk;
    for (int i = 0; i < want; ++i) {
        int best = -1;
        for (int b = 0; b < n_blk; ++b) {
            if (taken[b]) continue;
            if (best < 0 || scores[b] > scores[best]) best = b;   // strict > keeps the lower index on a tie
        }
        if (best < 0) break;
        taken[best] = true;
        s.blocks[s.n++] = best;
    }
    return s;
}

/// The blocks one token reads, per indexer head: `n_heads` rows of `scores`, each `n_blk` wide.  `out` gets
/// `n_heads` selections.  Returns false when the sizes do not fit.
inline bool select_all_heads(const float* scores, int n_heads, int n_blk, int k, BlockSel* out) {
    if (!scores || !out || n_heads <= 0 || n_blk <= 0) return false;
    for (int h = 0; h < n_heads; ++h) {
        out[h] = select_blocks(scores + (size_t)h * n_blk, n_blk, k);
        if (out[h].n == 0) return false;
    }
    return true;
}

}  // namespace strata::minimax
