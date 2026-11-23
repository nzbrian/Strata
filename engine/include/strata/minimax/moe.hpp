// include/strata/minimax/moe.hpp - MiniMax-M3.1-Flash's routed-expert maths: which experts a token gets, with what
// weight, the activation each of them runs, and how their rows are added back.
//
// Header-only, dependency-free and GPU-free on purpose: the parity test (tests/minimax/moe_test.cpp) runs this
// against a double-precision reference on any machine, and the CUDA path takes its numbers from the same functions.
//
// The formulas follow the model's own release implementation (llama.cpp at the pin in third_party/strata-llama:
// `llm_graph_context::build_moe_ffn` and `ggml_compute_forward_swiglu_oai_f32`), so a Strata answer and a llama.cpp
// answer on the same file differ only by arithmetic:
//
//   probs       = sigmoid(router_logits)                    (expert_gating_func 2 = sigmoid, not softmax)
//   selection   = probs + exp_probs_b                       (the learned bias moves WHICH experts are picked)
//   top-k       = the k highest selection scores.  llama.cpp sorts these with `ggml_argsort_top_k`, whose
//                 comparator looks at the score alone and runs under `std::sort` - unstable, so an exact tie may
//                 come out in any index order.  Strata pins the rule instead: equal scores take the lower expert
//                 index, so a token routes the same way on every backend and every thread count.
//   weight      = the selected expert's sigmoid prob - the bias selects, it does not weight
//   normalise   = w /= max(sum(w), kWeightsSumFloor)        (expert_weights_norm 1)
//   scale       = w *= 2.0                                  (expert_weights_scale)
//   activation  = swiglu_oai(gate, up) = gate * sigmoid(1.702 * gate) * (1 + clamp(up, -7, 7)), then down
//   combine     = out = sum_i w_i * expert_i(x) + shared(x)
//
// 128 routed experts with 4 per token plus one shared expert of the same width, as the file says; the sizes come
// from core::geometry_from_gguf, and only the array bounds below are fixed here.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strata::minimax {

/// The widest top-k a caller may ask for.  The model's own `expert_used_count` is 4; 16 is the ceiling the geometry
/// contract allows, so a Route stays a fixed-size value with no allocation on the token's hot path.
constexpr int kMaxExpertsPerToken = 16;

/// The most experts this scores on the stack.  The geometry contract stops at 512, so 512 covers every file that
/// passes the guard; `route` refuses anything wider rather than reading past its arrays.
constexpr int kMaxExperts = 512;

/// The floor under the weight sum: the smallest normal of F16, which is what the reference clamps the sum to, so a
/// token no expert wants divides by this and gets zeros instead of NaN.
constexpr float kWeightsSumFloor = 6.103515625e-5f;

/// swiglu_oai's constants, the same as gpt-oss and named in the graph builder.
constexpr float kSwigluAlpha = 1.702f;
constexpr float kSwigluLimit = 7.0f;

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// One token's routing: the experts it goes to, in selection order, with the weight each carries.
struct Route {
    int experts[kMaxExpertsPerToken];
    float weights[kMaxExpertsPerToken];
    int n;   ///< filled entries; 0 means the call was refused (see `route`)
};

/// Routes one token.  `logits` and `bias` (may be null) are `n_expert` wide.  `norm` and `scale` are the model's
/// `expert_weights_norm` and `expert_weights_scale`.  Equal selection scores keep the lower expert index.
inline Route route(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale) {
    Route r;
    r.n = 0;
    if (!logits || n_expert <= 0 || n_expert > kMaxExperts || k <= 0 || k > kMaxExpertsPerToken) return r;
    r.n = k;
    for (int s = 0; s < k; ++s) { r.experts[s] = -1; r.weights[s] = 0.0f; }
    float probs[kMaxExperts];
    for (int e = 0; e < n_expert; ++e) probs[e] = sigmoid(logits[e]);
    // pick the k best, one at a time, skipping what is already taken: the scores are not reordered, so equal scores
    // resolve to the lower expert index - the rule the CUDA top-k has to match
    bool taken[kMaxExperts];
    for (int e = 0; e < n_expert; ++e) taken[e] = false;
    for (int s = 0; s < k; ++s) {
        int best = -1;
        float best_score = 0.0f;
        for (int e = 0; e < n_expert; ++e) {
            if (taken[e]) continue;
            const float sc = probs[e] + (bias ? bias[e] : 0.0f);
            if (best < 0 || sc > best_score) { best = e; best_score = sc; }
        }
        if (best < 0) { r.n = s; break; }
        taken[best] = true;
        r.experts[s] = best;
        r.weights[s] = probs[best];   // the probability, not the biased selection score
    }
    if (norm) {
        float sum = 0.0f;
        for (int s = 0; s < r.n; ++s) sum += r.weights[s];
        if (sum < kWeightsSumFloor) sum = kWeightsSumFloor;
        for (int s = 0; s < r.n; ++s) r.weights[s] /= sum;
    }
    // 0 and 1 both mean "unscaled": llama-graph.cpp L2149 guards with `w_scale != 0.0f && w_scale != 1.0f` and
    // llama-hparams.h L124 defaults the scale to 0, so a missing key must leave the weights alone, not zero them.
    if (scale != 0.0f && scale != 1.0f)
        for (int s = 0; s < r.n; ++s) r.weights[s] *= scale;
    return r;
}

/// The routed and shared experts' activation: `gate` clamped from above, `up` clamped both ways, the gate through a
/// steepened sigmoid - and the (clamped) up term enters as `(1 + up)`, not as `up`.
///
/// Two things here are easy to get wrong and both fail silently:
///   - the gate's clamp is ONE-SIDED (`min`), the up-projection's is two-sided;
///   - the up term is `1 + clamp(up)`.  Drop the 1 and up = 0 zeroes the expert instead of passing the gate term
///     through, which pulls every routed and dense-lead row toward zero with no error anywhere.
/// Both backends agree, and this file follows them: ggml/src/ggml-cpu/ops.cpp L3371-3374 (`out_glu * (y + 1.f)`)
/// and ggml/src/ggml-cuda/unary.cuh L109-115 (`out_glu = out_glu * (1.0f + g)`).
inline float swiglu_oai(float gate, float up) {
    const float x = gate < kSwigluLimit ? gate : kSwigluLimit;
    const float y = up > kSwigluLimit ? kSwigluLimit : (up < -kSwigluLimit ? -kSwigluLimit : up);
    return x / (1.0f + std::exp(kSwigluAlpha * (-x))) * (1.0f + y);
}

/// One token's MoE output: the weighted rows of the selected experts plus the shared expert's row (`shared_row` may
/// be null), `n` floats each.  `expert_rows[s]` is the row of expert `r.experts[s]`.
inline void combine(const Route& r, const float* const* expert_rows, const float* shared_row, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        float acc = shared_row ? shared_row[i] : 0.0f;
        for (int s = 0; s < r.n; ++s) acc += r.weights[s] * expert_rows[s][i];
        out[i] = acc;
    }
}

}  // namespace strata::minimax
