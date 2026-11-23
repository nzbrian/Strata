// tests/minimax/indexer_test.cpp - MiniMax-M3.1-Flash's sparse-attention indexer (strata/minimax/indexer.hpp) with
// no model, no GPU and no ggml: the block score, the local force, the selection, and parity against a reference
// written here from the same source lines (models/minimax-m3.cpp at the vendored pin).
//
//   * the block score is the MAX of the block's raw query-key dots - no 1/sqrt(dim), nothing else;
//   * a position with no key (-INFINITY) cannot carry a block into the selection, and a whole masked block stays out;
//   * the block holding the query's position (and the `local_blocks` before it) is forced in by +1e30, which is so
//     large that it erases the block's own score in F32 - so forced blocks tie, and their order is the tie rule's;
//   * a block size that does not divide the written positions rounds up, and the padding does not vote;
//   * equal block scores take the lower block index - Strata's rule; llama.cpp's `ggml_top_k` compares the score
//     alone under an unstable partial sort, so a tie there has no defined order and parity is checked as a set.
#include "strata/minimax/indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace mm = strata::minimax;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
struct Rng {
    unsigned long long s = 0x2545F4914F6CDD1Dull;
    unsigned long long next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float between(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() % 1000000) / 1000000.0f; }
};

std::vector<float> dots_to_scores(const std::vector<float>& q, const std::vector<float>& keys,
                                 const std::vector<float>& mask, int n_pos, int dim, int blk,
                                 long long pos, int local) {
    const int nb = (n_pos + blk - 1) / blk;
    std::vector<float> s(nb, -INFINITY);
    for (int p = 0; p < n_pos; ++p) {
        if (mask[p] == -INFINITY) continue;
        double acc = 0.0;
        for (int d = 0; d < dim; ++d) acc += (double)q[d] * keys[(size_t)p * dim + d];
        const float v = (float)(acc + mask[p]);
        float& b = s[p / blk];
        if (v > b) b = v;
    }
    const long long own = pos / blk;
    for (int l = 0; l < local && own - l >= 0; ++l) {
        if (own - l < nb) s[own - l] += mm::kLocalForceBias;
    }
    return s;
}

bool is_set(const mm::BlockSel& s) {
    std::vector<int> v(s.blocks, s.blocks + s.n);
    std::sort(v.begin(), v.end());
    for (size_t i = 1; i < v.size(); ++i) if (v[i] == v[i - 1]) return false;
    return true;
}
}  // namespace

int main() {
    std::printf("indexer_test (MiniMax-M3.1-Flash sparse attention)\n");
    {
        // block scores are the max of the block's raw dots: q={1,0} reads the first coordinate back
        const std::vector<float> q = {1.0f, 0.0f};
        const std::vector<float> keys = {1, 0,  2, 0,      // block 0: 1, 2   -> max 2
                                         5, 0,  0, 0,      // block 1: 5, 0   -> max 5
                                         3, 0,  4, 0};     // block 2: 3, 4   -> max 4
        std::vector<float> mask(6, 0.0f), out(3, 0.0f);
        mm::block_scores(q.data(), keys.data(), mask.data(), 6, 2, 2, /*pos*/ 4, /*local*/ 0, out.data());
        check(out[0] == 2.0f && out[1] == 5.0f && out[2] == 4.0f, "a block scores the max of its raw dots, unscaled");
        mm::BlockSel s = mm::select_blocks(out.data(), 3, 2);
        check(s.n == 2 && s.blocks[0] == 1 && s.blocks[1] == 2, "the two best blocks come out highest score first");
    }
    {
        // no 1/sqrt: {3,4} . {3,4} is 25, not 25/sqrt(2)
        const float q[2] = {3.0f, 4.0f}, k[2] = {3.0f, 4.0f};
        check(std::fabs(mm::dot(q, k, 2) - 25.0f) < 1e-6f, "the indexer dot is raw: 25, no 1/sqrt(dim)");
    }
    {
        // the local force: block 0 has the worst score in the file and is read anyway
        const std::vector<float> q = {1.0f, 0.0f};
        const std::vector<float> keys = {0, 0, 0, 0,   9, 0, 9, 0,   8, 0, 8, 0};
        std::vector<float> mask(6, 0.0f), out(3, 0.0f), out2(3, 0.0f);
        mm::block_scores(q.data(), keys.data(), mask.data(), 6, 2, 2, /*pos*/ 1, /*local*/ 1, out.data());
        check(out[0] > mm::kLocalForceBias * 0.9f, "the query's own block is lifted by the 1e30 force");
        mm::BlockSel s = mm::select_blocks(out.data(), 3, 2);
        check(s.blocks[0] == 0 && s.blocks[1] == 1, "so it is read even though its score is the lowest");
        mm::block_scores(q.data(), keys.data(), mask.data(), 6, 2, 2, /*pos*/ 1, /*local*/ 0, out2.data());
        mm::BlockSel s0 = mm::select_blocks(out2.data(), 3, 2);
        check(s0.blocks[0] == 1 && s0.blocks[1] == 2, "with local_blocks 0 the same query ignores block 0");
    }
    {
        // local_blocks 2 forces the block before the query's too (M3 ships 1; the rule is the file's)
        const std::vector<float> q = {1.0f, 0.0f};
        const std::vector<float> keys = {1, 0, 1, 0,   2, 0, 2, 0,   3, 0, 3, 0};
        std::vector<float> mask(6, 0.0f), out(3, 0.0f);
        mm::block_scores(q.data(), keys.data(), mask.data(), 6, 2, 2, /*pos*/ 3, /*local*/ 2, out.data());
        mm::BlockSel s = mm::select_blocks(out.data(), 3, 3);
        check(out[0] == out[1], "the 1e30 force erases the score in F32: the two forced blocks tie (eps at 1e30 is ~1e23)");
        check(s.blocks[0] == 0 && s.blocks[1] == 1, "so the tie rule orders them by block index, older first");
        check(s.blocks[2] == 2, "and the unforced block follows on its own score");
        check(out[2] == 3.0f, "which the force leaves untouched");
    }
    {
        // a hole in the past: a masked position cannot speak for its block, a fully masked block stays out
        const std::vector<float> q = {1.0f, 0.0f};
        const std::vector<float> keys = {7, 0, 0, 0,   9, 0, 9, 0,   1, 0, 1, 0};
        std::vector<float> mask = {0.0f, -INFINITY, 0.0f, 0.0f, -INFINITY, -INFINITY};
        std::vector<float> out(3, 0.0f);
        mm::block_scores(q.data(), keys.data(), mask.data(), 6, 2, 2, /*pos*/ 5, /*local*/ 0, out.data());
        check(out[0] == 7.0f, "a masked position does not lower its block's score");
        check(out[2] == -INFINITY, "a block with no written position scores -INFINITY");
        mm::BlockSel s = mm::select_blocks(out.data(), 3, 2);
        check(s.n == 2 && s.blocks[0] == 1 && s.blocks[1] == 0, "and it is the last thing a query reads");
    }
    {
        // the tail: 5 written positions with a block of 2 make 3 blocks, the last with one vote
        const std::vector<float> q = {1.0f, 0.0f};
        const std::vector<float> keys = {1, 0, 2, 0, 3, 0, 4, 0, 6, 0};
        std::vector<float> mask(5, 0.0f), out(3, 0.0f);
        mm::block_scores(q.data(), keys.data(), mask.data(), 5, 2, 2, /*pos*/ 4, /*local*/ 0, out.data());
        check(out[2] == 6.0f, "the odd tail block is scored from the one position it holds");
    }
    {
        // ties: Strata's rule, written down because llama.cpp's partial sort does not define it
        const std::vector<float> q = {1.0f, 0.0f};
        const std::vector<float> keys = {5, 0, 5, 0,   1, 0, 1, 0,   5, 0, 5, 0};
        std::vector<float> mask(6, 0.0f), out(3, 0.0f);
        mm::block_scores(q.data(), keys.data(), mask.data(), 6, 2, 2, /*pos*/ 0, /*local*/ 0, out.data());
        mm::BlockSel s = mm::select_blocks(out.data(), 3, 3);
        check(s.blocks[0] == 0 && s.blocks[1] == 2 && s.blocks[2] == 1,
              "three equal scores are taken in block order (Strata's rule, not llama.cpp's)");
    }
    {
        // heads choose for themselves: the same blocks, different queries
        const std::vector<float> scores = {/*head 0*/ 1.0f, 9.0f, 5.0f, /*head 1*/ 8.0f, 2.0f, 7.0f};
        mm::BlockSel sel[2];
        check(mm::select_all_heads(scores.data(), 2, 3, 2, sel), "two heads select from one score table");
        check(sel[0].blocks[0] == 1 && sel[0].blocks[1] == 2 && sel[1].blocks[0] == 0 && sel[1].blocks[1] == 2,
              "and each head keeps its own order");
    }
    {
        // parity with the reference over the shape the model uses: 4 heads of 128, blocks of 128
        Rng g;
        const int dim = 128, blk = 128, K = 16;
        int bad_set = 0, bad_order = 0, tight = 0;
        double worst = 0.0;
        for (int t = 0; t < 600; ++t) {
            const int n_pos = 1 + (int)(g.next() % 900);
            std::vector<float> q(dim), keys((size_t)n_pos * dim), mask(n_pos, 0.0f);
            for (int d = 0; d < dim; ++d) q[d] = g.between(-1.0f, 1.0f);
            for (auto& v : keys) v = g.between(-1.0f, 1.0f);
            for (int p = 0; p < n_pos; ++p) if ((g.next() % 9) == 0) mask[p] = -INFINITY;
            const long long pos = (long long)(g.next() % (unsigned long long)n_pos);
            std::vector<float> mine((size_t)n_pos / blk + 1, 0.0f);
            const int nb = (n_pos + blk - 1) / blk;
            mine.assign(nb, 0.0f);
            mm::block_scores(q.data(), keys.data(), mask.data(), n_pos, dim, blk, pos, 1, mine.data());
            std::vector<float> ref = dots_to_scores(q, keys, mask, n_pos, dim, blk, pos, 1);
            for (int b = 0; b < nb; ++b)
                if (std::isfinite(ref[b])) worst = std::max(worst, std::fabs((double)mine[b] - ref[b]));
            mm::BlockSel s = mm::select_blocks(mine.data(), nb, K > nb ? nb : K);
            if (!is_set(s)) ++bad_set;   // never select the same block twice
            // the reference's own order: sort by score desc, ties by lower index
            std::vector<int> order(nb);
            for (int b = 0; b < nb; ++b) order[b] = b;
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int c) { return ref[a] > ref[c]; });
            bool strict = true;
            for (int i = 0; i + 1 < (int)s.n && i + 1 < nb; ++i) if (ref[order[i]] == ref[order[i + 1]]) strict = false;
            if (!strict) { ++tight; continue; }
            for (int i = 0; i < s.n; ++i) {
                std::vector<int> ms(s.blocks, s.blocks + s.n), rs(order.begin(), order.begin() + s.n);
                if (std::find(rs.begin(), rs.end(), s.blocks[i]) == rs.end()) { ++bad_set; break; }
                if (ms[i] != rs[i]) { ++bad_order; break; }
            }
        }
        check(bad_set == 0, "600 random queries select the reference's blocks, each once");
        check(bad_order == 0, "and in its order whenever the scores are strictly ordered");
        std::printf("    queries: 600, score-tight skipped: %d, worst absolute block-score difference: %.3e\n",
                    tight, worst);
    }
    {
        // the refusals
        std::vector<float> sc(8, 1.0f);
        check(mm::select_blocks(sc.data(), 8, 33).n == 0, "selecting more than the ceiling is refused");
        check(mm::select_blocks(nullptr, 8, 4).n == 0, "no scores is refused");
        check(mm::select_blocks(sc.data(), 0, 4).n == 0, "an empty cache is refused");
        std::vector<float> out(4, 7.0f), q(2, 1.0f);
        mm::block_scores(nullptr, q.data(), nullptr, 4, 2, 2, 0, 1, out.data());
        check(out[0] == 7.0f && out[1] == 7.0f && out[2] == 7.0f && out[3] == 7.0f,
              "a null query writes nothing: the caller's buffer is left exactly as it was");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASSED");
    return g_fail ? 1 : 0;
}
