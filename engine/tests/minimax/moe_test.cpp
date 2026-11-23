// tests/minimax/moe_test.cpp - MiniMax-M3.1-Flash's routed-expert maths (strata/minimax/moe.hpp) with no model, no
// GPU and no ggml: known answers, the edges the file's own numbers create, and parity against a reference
// written here from the same source lines (llama.cpp's build_moe_ffn and swiglu_oai at the vendored pin, weights in
// double, selection in F32).
//
//   * the learned bias moves WHICH experts a token gets, never how much each one weighs;
//   * after the normalise and the scale the weights sum to expert_weights_scale (2.0), not to 1;
//   * a token every expert refuses still gets finite weights (the F16 floor under the sum, not a NaN);
//   * equal selection scores go to the lower expert index - Strata's own rule, written down because llama.cpp's
//     argsort compares the score only under an unstable sort and leaves a tie in any order;
//   * the selection runs on F32 scores, as the model's own code does, so a selection made in double can order the
//     third and fourth expert differently when their scores are within a float epsilon;
//   * swiglu_oai clamps the gate from above only, the up-projection both ways, and the up term enters as (1 + up).
#include "strata/minimax/moe.hpp"

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
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    unsigned long long next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float between(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() % 1000000) / 1000000.0f; }
};

// ---- the reference.  The weights are carried in double; the SELECTION runs on F32 scores, because that is what the
// model's own code does (build_moe_ffn: probs = ggml_sigmoid on F32, then ggml_argsort_top_k over the F32 scores).
// A double selection is not "more correct" here - it is a different basis, and where two scores land within a float
// epsilon of each other the third and fourth expert can swap.  Comparing on the model's basis is the only comparison
// that means anything, and a CUDA top-k has to select on that basis with the same tie rule.
// The tight-token count printed below measures THESE FIXTURES (uniform logits in [-12, 12]).  How tightly the real
// model's router scores crowd is NOT measured here - no real hidden state was run, so no claim is made about it.
double ref_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
float ref_sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void ref_route(const std::vector<float>& logits, const std::vector<float>& bias, int k, bool norm, float scale,
               std::vector<int>& idx, std::vector<double>& w, std::vector<float>* sel_out = nullptr) {
    const int n = static_cast<int>(logits.size());
    std::vector<float> probs(n), sel(n);
    for (int e = 0; e < n; ++e) {
        probs[e] = ref_sigmoid_f(logits[e]);
        sel[e] = probs[e] + (bias.empty() ? 0.0f : bias[e]);
    }
    if (sel_out) *sel_out = sel;
    std::vector<bool> used(n, false);
    idx.assign(k, -1);
    w.assign(k, 0.0);
    for (int s = 0; s < k; ++s) {
        int best = -1;
        for (int e = 0; e < n; ++e) {
            if (used[e]) continue;
            if (best < 0 || sel[e] > sel[best]) best = e;   // strict > keeps the lower index on a tie
        }
        if (best < 0) break;
        used[best] = true;
        idx[s] = best;
        w[s] = probs[best];
    }
    if (norm) {
        double sum = 0.0;
        for (double x : w) sum += x;
        if (sum < mm::kWeightsSumFloor) sum = mm::kWeightsSumFloor;
        for (double& x : w) x /= sum;
    }
    if (scale != 1.0f)
        for (double& x : w) x *= scale;
}

// Transcribed from the reference in double, apart from the header: x = min(gate, limit), y = clamp(up, -limit,
// limit), out = x / (1 + exp(-alpha*x)) * (1 + y).  Sources: ggml/src/ggml-cpu/ops.cpp L3371-3374 and
// ggml/src/ggml-cuda/unary.cuh L109-115.  That `(1 + y)` is what BOTH this reference and the header got wrong once -
// parity between them proved nothing - so the absolute anchors in the swiglu block are computed from those two
// source lines and not from either implementation.
double ref_swiglu_oai(double gate, double up) {
    const double x = std::min(gate, static_cast<double>(mm::kSwigluLimit));
    const double y = std::clamp(up, -static_cast<double>(mm::kSwigluLimit), static_cast<double>(mm::kSwigluLimit));
    return x / (1.0 + std::exp(mm::kSwigluAlpha * -x)) * (1.0 + y);
}

bool same_weights(const mm::Route& r, const std::vector<double>& w, double tol) {
    for (int s = 0; s < r.n; ++s)
        if (std::fabs(r.weights[s] - w[s]) > tol * (std::fabs(w[s]) + 1e-9)) return false;
    return true;
}
}  // namespace

int main() {
    std::printf("moe_test (MiniMax-M3.1-Flash routed experts)\n");
    {
        // the model's own numbers: 128 experts, 4 per token, normalise, scale 2.0, no bias
        std::vector<float> logits(128, 0.0f), bias;
        mm::Route r = mm::route(logits.data(), nullptr, 128, 4, true, 2.0f);
        check(r.n == 4 && r.experts[0] == 0 && r.experts[3] == 3, "a flat router takes the first four experts by index");
        double sum = 0.0;
        bool each_half = true;
        for (int s = 0; s < r.n; ++s) { sum += r.weights[s]; if (std::fabs(r.weights[s] - 0.5f) > 1e-6) each_half = false; }
        check(each_half, "four equal experts of 0.5 each weigh 0.5 (0.5 / 2.0 sum, then the scale 2.0)");
        check(std::fabs(sum - 2.0) < 1e-5, "and the weights sum to expert_weights_scale, not to 1");
    }
    {
        // the bias decides who is picked, and does not ride along into the weight
        std::vector<float> logits = {1.0f, 0.99f}, bias = {0.0f, 0.5f};
        mm::Route r = mm::route(logits.data(), bias.data(), 2, 1, false, 1.0f);
        check(r.n == 1 && r.experts[0] == 1, "the bias lifts the slightly weaker expert into the top-k");
        check(std::fabs(r.weights[0] - mm::sigmoid(0.99f)) < 1e-7, "and its weight is its own sigmoid, bias excluded");
    }
    {
        // scale 0 means UNSCALED to the reference (llama-graph.cpp L2149 tests `w_scale != 0.0f && w_scale != 1.0f`,
        // and llama-hparams.h L124 defaults the scale to 0).  Treating 0 as a real factor zeroes the routed side and
        // leaves the shared expert alone - a silent, plausible-looking model.
        std::vector<float> logits = {1.0f, 0.9f, 0.8f, 0.7f}, bias(4, 0.0f);
        mm::Route r0 = mm::route(logits.data(), bias.data(), 4, 4, true, 0.0f);
        double sum0 = 0.0;
        for (int s = 0; s < r0.n; ++s) sum0 += r0.weights[s];
        check(std::fabs(sum0 - 1.0) < 1e-5, "a scale of 0 leaves the normalised weights summing to 1, not to 0");
        mm::Route r1 = mm::route(logits.data(), bias.data(), 4, 4, true, 1.0f);
        check(std::fabs(r1.weights[0] - r0.weights[0]) < 1e-7f, "and 0 and 1 mean the same thing, as they do there");
    }
    {
        // a token no expert wants: the sum floor, not a NaN
        std::vector<float> logits(128, -1000.0f);
        mm::Route r = mm::route(logits.data(), nullptr, 128, 4, true, 2.0f);
        bool finite = true;
        for (int s = 0; s < r.n; ++s) if (!std::isfinite(r.weights[s])) finite = false;
        check(finite, "an all-dead token still gives finite weights");
        check(std::fabs(r.weights[0] - r.weights[3]) < 1e-9, "and they stay equal to each other");
    }
    {
        // ties: equal scores take the lower index, the rule a parallel top-k has to reproduce
        std::vector<float> logits(8, 0.25f);
        mm::Route r = mm::route(logits.data(), nullptr, 8, 3, false, 1.0f);
        check(r.experts[0] == 0 && r.experts[1] == 1 && r.experts[2] == 2, "equal selection scores keep the lower expert index");
    }
    {
        // parity with the reference: selection on the same F32 scores, so the expert ORDER must agree exactly; the
        // reference carries the weights further, which is why those get a tolerance instead of equality.  A token
        // whose four chosen scores and best reject are all within a hair of each other is counted and skipped -
        // there a different contraction order may legally swap two experts.
        Rng g;
        int bad_idx = 0, bad_w = 0, tight = 0;
        double worst = 0.0;
        for (int t = 0; t < 4000; ++t) {
            const int ne = 128;
            std::vector<float> logits(ne), bias(ne);
            for (int e = 0; e < ne; ++e) { logits[e] = g.between(-12.0f, 12.0f); bias[e] = g.between(-1.0f, 1.0f); }
            const bool norm = (t & 1) != 0;
            const float scale = norm ? 2.0f : 1.0f;
            const bool use_bias = (t % 3) != 0;
            mm::Route r = mm::route(logits.data(), use_bias ? bias.data() : nullptr, ne, 4, norm, scale);
            std::vector<int> idx;
            std::vector<double> w;
            std::vector<float> sel;
            ref_route(logits, use_bias ? bias : std::vector<float>(), 4, norm, scale, idx, w, &sel);
            std::vector<float> srt = sel;
            std::sort(srt.begin(), srt.end(), std::greater<float>());
            float tightest = srt[3] - srt[4];
            for (int s = 0; s < 3; ++s) tightest = std::min(tightest, srt[s] - srt[s + 1]);
            if (tightest < 1e-9f) { ++tight; continue; }
            // an EXACT tie inside the top-k: llama.cpp's sort leaves its order undefined, so only the set is
            // comparable there.  With strictly ordered scores the order has to match exactly.
            bool tie = false;
            for (int s = 0; s + 1 < 4; ++s) if (srt[s] == srt[s + 1]) tie = true;
            std::vector<int> mine(r.experts, r.experts + 4), ref = idx;
            if (tie) { std::sort(mine.begin(), mine.end()); std::sort(ref.begin(), ref.end()); }
            if (mine != ref) { ++bad_idx; continue; }
            // weights are checked for EVERY token whose set matches.  They used to be checked only inside the `tie`
            // branch below, and exact F32 ties across 128 random logits essentially never happen - that comparison
            // could not fail, which is the opposite of what a parity test is for.
            for (int s = 0; s < 4; ++s) worst = std::max(worst, std::fabs(r.weights[s] - w[s]));
            if (!same_weights(r, w, 1e-5)) ++bad_w;
            if (!tie) {
                for (int s = 0; s < 4; ++s) if (r.experts[s] != idx[s]) { ++bad_idx; break; }
            }
        }
        check(bad_idx == 0, "random tokens pick the same experts as the F32 reference (its order, unless it ties)");
        check(bad_w == 0, "and their weights agree with it to 1e-5 relative");
        std::printf("    tokens compared: 4000, score-tight tokens skipped: %d, worst absolute weight difference: %.3e\n",
                    tight, worst);
    }
    {
        // swiglu_oai: the gate clamps on one side, the up-projection on both
        check(std::fabs(mm::swiglu_oai(2.0f, 3.0f) - ref_swiglu_oai(2.0, 3.0)) < 1e-6, "swiglu_oai(2, 3) matches the reference");
        check(std::fabs(mm::swiglu_oai(100.0f, 1.0f) - ref_swiglu_oai(7.0, 1.0)) < 1e-6, "the gate clamps at +7");
        check(std::fabs(mm::swiglu_oai(-100.0f, 1.0f)) < 1e-6, "a deeply negative gate gives ~0, not -7");
        check(std::fabs(mm::swiglu_oai(1.0f, -100.0f) - ref_swiglu_oai(1.0, -7.0)) < 1e-6, "the up-projection clamps both ways");
        // Absolute anchors, computed in double from the two reference source lines - not from this header, not from
        // ref_swiglu_oai - so they fail if either one drifts.  (2, 0) is the check that catches a dropped `+ 1`:
        // with up = 0 the reference passes the gate term through (1.935658623); without the 1 it returns 0.
        check(std::fabs(mm::swiglu_oai(2.0f, 0.0f) - 1.935658623f) < 1e-6f, "swiglu_oai(2, 0) = 1.935658623 - up 0 passes the gate through");
        check(std::fabs(mm::swiglu_oai(1.0f, 1.0f) - 1.691591532f) < 1e-6f, "swiglu_oai(1, 1) = 1.691591532");
        check(std::fabs(mm::swiglu_oai(10.0f, -7.0f) + 41.999718770f) < 1e-4f, "swiglu_oai(10, -7) = -41.999718770 - gate one-sided, up both");
        check(std::fabs(mm::swiglu_oai(7.0f, 7.0f) - 55.999625026f) < 1e-4f, "swiglu_oai(7, 7) = 55.999625026");
        check(mm::swiglu_oai(0.0f, 5.0f) == 0.0f, "a gate of 0 gives 0 whatever up says");
        check(mm::swiglu_oai(-10.0f, 0.0f) < 0.0f && mm::swiglu_oai(-10.0f, 0.0f) > -1e-5f,
              "a large negative gate stays small and negative instead of clamping to -7");
        Rng g;
        double worst = 0.0;
        for (int t = 0; t < 20000; ++t) {
            const float a = g.between(-20.0f, 20.0f), b = g.between(-20.0f, 20.0f);
            worst = std::max(worst, std::fabs(mm::swiglu_oai(a, b) - ref_swiglu_oai(a, b)));
        }
        check(worst < 1e-5, "20000 random pairs agree with the reference within 1e-5");
        std::printf("    worst absolute activation difference: %.3e\n", worst);
    }
    {
        // combine: the shared expert rides along once, unweighted
        mm::Route r;
        r.n = 2;
        r.experts[0] = 5; r.weights[0] = 0.75f;
        r.experts[1] = 9; r.weights[1] = 0.25f;
        const float e5[3] = {2.0f, 0.0f, -1.0f}, e9[3] = {0.0f, 4.0f, 1.0f}, sh[3] = {1.0f, 1.0f, 1.0f};
        const float* rows[2] = {e5, e9};
        float out[3] = {0, 0, 0};
        mm::combine(r, rows, sh, 3, out);
        check(std::fabs(out[0] - 2.5f) < 1e-6 && std::fabs(out[1] - 2.0f) < 1e-6 && std::fabs(out[2] - 0.5f) < 1e-6,
              "out = 0.75*expert5 + 0.25*expert9 + shared, elementwise (1.0 - 0.75 + 0.25 = 0.5)");
        mm::combine(r, rows, nullptr, 3, out);
        check(std::fabs(out[1] - 1.0f) < 1e-6, "a layer with no shared expert (there is none in M3) still adds up");
    }
    {
        // the refusals: a Route with n 0, never a read past the arrays
        std::vector<float> logits(600, 0.1f);
        check(mm::route(logits.data(), nullptr, 600, 4, true, 2.0f).n == 0, "more experts than the contract allows is refused");
        check(mm::route(logits.data(), nullptr, 128, 0, true, 2.0f).n == 0, "top-k 0 is refused");
        check(mm::route(logits.data(), nullptr, 128, 17, true, 2.0f).n == 0, "top-k above the ceiling is refused");
        check(mm::route(nullptr, nullptr, 128, 4, true, 2.0f).n == 0, "no logits is refused");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASSED");
    return g_fail ? 1 : 0;
}
