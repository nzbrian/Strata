# MiniMax-M3.1-Flash: where it stands and what a Strata run still needs

Not a promise and not a supported feature. This records what was measured on 2026-11-18 on one machine, and the
work that remains before Strata can run the model. Nothing here has been benchmarked against another engine.

## The model, from the file

`AtomicChat/MiniMax-M3.1-Flash-GGUF`, `general.architecture = minimax-m3`, quant `AD-IQ2_XXS` (3 shards + mmproj).

| Fact | Value | How it was obtained |
| --- | --- | --- |
| Parameters | 426.17 B | `llama_model_n_params` on the loaded model |
| Weight bytes | 104.45 GiB (95.89 GiB routed experts = 91.8% of bytes, 96.9% of parameters) | summed `n_bytes` over all 1,176 tensors |
| Quant types in the experts | IQ1_M, IQ2_XXS, IQ2_XS; attention Q4_K/Q8_0/F32, dense FFN Q6_K, shared Q5_K, output Q4_K | the tensor table |
| One expert slot (gate+up+down, one layer) | 13.46 MiB | 95.89 GiB / (57 layers x 128 experts) |
| Expert bytes touched per token at top-4 + 1 shared | 3.75 GiB, with no reuse | 5 x 57 x 13.46 MiB — arithmetic, not a measurement |
| Geometry | 60 layers (3 dense leading), hidden 6144, 64 Q / 4 KV heads at 128, RoPE over 64 of 128 dims, 128 experts top-4 + 1 shared at 3072, indexer 4 heads x 128, top-16 of 128-position blocks, 1 local block, vocab 200,064 | read by `model_family_test --real` against the file |
| Router constants | sigmoid gating (`expert_gating_func = 2`), `expert_weights_norm = 1`, `expert_weights_scale = 2.0`, weight-sum floor 6.103515625e-5, swiglu alpha 1.702 / limit 7.0 | the file's metadata and the reference source |

## Measured on this machine (2026-11-18)

RTX 5060 Ti 16 GB + RTX 5060 Ti 16 GB + RTX 3060 12 GB, Xeon E5-2630 v2 (no AVX2, no FMA), 247 GB RAM, model on
the same NVMe as everything else. Runner: `tools/m3ref/m3run`, which links the vendored llama.cpp fork — **no
Strata kernel ran in any of these numbers.**

| Configuration | Result |
| --- | --- |
| Attention, shared expert, router, indexer and KV on one 5060 Ti; routed experts in RAM; ctx 4096, batch 128, 8 threads | load **32.0 s**, peak RSS **106,720 MiB**, prefill 2.08 tok/s (7 tokens, mostly fixed cost), decode **2.42 tok/s**, 24 tokens produced, exit 0 |
| Same, but all layers on CPU, ctx 256 | load 161.3 s, peak RSS 68,182 MiB; the graph built, then failed for VRAM because another process held the cards |

Two build facts that cost time and are easy to repeat:

- The fork compiles `ggml-cpu` with `-mfma -mavx2` **even under `GGML_NATIVE=OFF`**, which is `SIGILL` on this CPU.
  They must be disabled one by one: `-DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF` (AVX, F16C, SSE4.2 stay on).
- `llama_model_params.tensor_buft_overrides` takes a **bare tensor-name regex**. The `=CPU` suffix is llama.cpp's
  command-line syntax; through the API it matches nothing and the loader asks the card for all 104 GiB.

The completion from that run is the base model with **no chat template**, greedy, so it degenerates into markup.
That is expected and is a tokenizer/template gap, not a weight or kernel gap.

## What is in Strata now

- The family is recognised, its metadata is guarded, its geometry is read and its 1,176 tensor names check against
  a table of its own; it is still **refused for execution** with a message that says so (`model_family.hpp`).
- The router and the indexer are implemented header-only and tested, pinned to the reference source rather than to
  a paraphrase: `include/strata/minimax/moe.hpp`, `include/strata/minimax/indexer.hpp`.
- The `minimax-m2` pre-tokenizer rule is in `tools/strata_tokenizer.py`, compared string-for-string with the
  vendored source, and the real 200,064-token vocabulary round-trips (`decode(encode(s)) == s`).
- 144/144 tests in the CPU-only build; `layout: ok (1176 tensors checked)` against the real file.

## What a Strata-native run still needs, in order

1. **An engine path with runtime shapes.** The compiled Flash-Next contract rejects M3 on four counts
   (`head_dim` 128 vs 256, `n_embd` 6144 vs 1024..4096, `n_ff` 3072 vs 256..1100, indexer pooling block 128 vs 4).
   `src/qwen35` is the precedent: its own kernels, no `static_assert`ed geometry.
2. **Kernels**: the indexer (dot + max-pool + top-k over blocks + the local force), the sparse attention that reads
   only the selected blocks, the indexer-key KV cache, and the expert GEMM/GEMV for IQ1_M / IQ2_XXS / IQ2_XS with
   `swiglu_oai` fused, including `(1 + clamp(up))`.
3. **Expert cache and file tier at M3 slot sizes** — 128 x 57 slots of 13.46 MiB. `ExpertCache::open_sized()` already
   takes per-slot byte counts and accepts up to 65,535 experts, so this is sizing and policy work, not a new cache.
4. **Tokenizer in C++**: GPT-2 BPE with `pre = minimax-m2` and the 11,629-character chat template with MiniMax's own
   literal special tokens (`]<]minimax[>[`, `]~!b[`, `]~b]`; bos 200019, eos 200020). `llama_chat_apply_template`
   cannot help — it has no jinja parser.
5. **Product wiring** so `strata launch` and the app recognise the family and keep refusing it until 1-4 exist.
6. **The measured number that decides whether it is worth shipping here**: 3.75 GiB of expert weights per token has
   to come from RAM at ~2.4 tok/s on this box. More cards do not fix that; RAM bandwidth and expert reuse do.

## Not measured

No Strata kernel ran. No perplexity, no quality comparison, no long-context run, no vision (the mmproj is downloaded
but unread), no batch > 1, no KV cache type experiments, and no run with the experts on GPU. The 3.75 GiB-per-token
figure is arithmetic on tensor sizes with zero reuse, not a speed and not a prediction.

## Continue here (2026-11-18, late)

Step 0, the blocker: `m3_native` and the llama.cpp reference disagree on the next token (200058 vs 53390) for
`--tokens 20768,841,14997,894,275,26225,46` ("Write one sentence about the Moon.", no chat template). `m3_native`
and an independent numpy+gguf-py pass now agree on layer 0 to ~3.5e-3 relative (sum 180.831 vs 181.052) - too big
for float32-vs-double accumulation, too small to explain the disjoint top-5, so there is a third divergence inside
layer 0. Localise it by fingerprinting the sub-blocks in both paths: after attention-before-FFN, and after FFN.
The dump hook is `--dump N` in `tools/m3ref/m3_native.cpp` (prints sum, sumsq, x0, x1, x_last of the last token
after layer N); the numpy path is /tmp/m3_np.py on this machine (reshape GGUF 2-D tensors as [ne1][ne0], dim 0
fastest - reshaping as [ne0][ne1] scrambles them and cost an hour).

Then, in order, each step ending in a commit and a check that can fail:

1. Acceptance harness: `m3_native` top-1 and top-5 logits against `m3run` for a fixed prompt, golden file in the
   repo. Nothing after this is believed until this passes.
2. `src/minimax/m3_engine.cu` + `include/strata/minimax/engine.hpp`, `strata::minimax::run()`, mirroring
   `src/qwen35`: weights uploaded still quantised, `--serve` protocol (READY/GEN/T/DONE/STOP/PP), `--tokens`.
   The family switch is `src/program/generate.cpp` "THE FAMILY SWITCH"; `model_family_has_engine()` stays false
   until the acceptance test passes on GPU.
3. Kernels, one at a time, each with a parity test against `include/strata/minimax/forward.hpp`: fused
   rms+add_rms; partial RoPE with QK-norm; projections through `strata::kernels::iq_mmvq(ggml_type, ...)` (it is
   family-neutral already); GQA causal attention over the KV cache at head_dim 128 with no output gate; the router
   (F32 selection, lower-index tie rule, weight sum = scale) and the expert GEMV with `swiglu_oai` fused including
   `(1 + clamp(up))`; then MSA.
4. MSA needs new kernels, not QSA's: block 128 against QSA's 4, top-16 blocks against 2048 cells. Indexer dot +
   max-pool + the `+1e30` local bias + top-k + gather, parity against `include/strata/minimax/indexer.hpp`.
5. Expert cache at M3 slot sizes - `ExpertCache::open_sized()` already takes per-slot byte counts (13.46 MiB here)
   and accepts up to 65,535 experts, so this is policy and sizing, not a new cache.
6. Tokenizer and template on the host side (`tools/strata_tokenizer.py` already handles `pre = minimax-m2` and
   round-trips the real vocabulary; the 11,629-character jinja template is the open piece - `llama_chat_apply_template`
   cannot help, it has no jinja parser), then `--family`, the model list, capabilities, docs.

Measured baseline to beat, from the vendored runner on one 5060 Ti with experts in RAM: load 32.0 s, peak RSS
106,720 MiB, decode 2.42 tok/s. `m3_native` on CPU is 583.6 s for one token's logits and 30.5 GiB resident.

### Step 0, ruled out (2026-11-18, later)

- Tokenisation is identical: libllama and `tools/strata_tokenizer.py` both give `20768,841,14997,894,275,26225,46`.
- Layer 0 of `m3_native` matches an independent numpy+gguf-py pass to ~2e-7 (sum 180.831352 vs 180.831321) once the
  script reads `rope.freq_base` (**10,000,000**, not 5e6) and `layer_norm_rms_epsilon` from the file. So the dense
  layer recipe - RMS, QK-norm before partial RoPE, GQA causal at 1/sqrt(128), dense `swiglu_oai` - is right.
- `weight_before_ffn` (routing weights applied to the expert input) is **Llama4 only**: `llama-graph.cpp:2020`.
  MiniMax weights the expert output, as `combine()` does. Not the cause.
- `swiglu_clamp_shexp` is inside `case LLM_FFN_SILU` only; M3 uses `LLM_FFN_SWIGLU_OAI_MOE` for dense, routed and
  shared alike (`minimax-m3.cpp:546,555,569`), and the file carries no swiglu/clamp keys, so alpha 1.702 and limit
  7.0 are the defaults everywhere. Not the cause.
- No `rope.type`, `rope.scale`, `sliding_window`, softcap or embedding scale in the file. Not the cause.
- For this 7-token prompt every position is inside indexer block 0, so MSA selection cannot change the attention
  output here - the divergence is not the block choice either.

Next single action: numpy cross-check of the **first sparse layer (blk.3)** - router, four routed slots, shared
expert - against `m3_native --dump 3`. If native and numpy agree there too, the divergence is a step the recipe is
missing, not an implementation error, and the reference graph has to be diffed against `forward.hpp` op by op
(`msa->bias`, `pos_slot`, and what the no-flash-attn MSA fallback actually computes are the live candidates).

### Step 0, also ruled out

- Expert-slot dequant is **byte-exact against ggml**: one full slot of `blk.3.ffn_gate_exps.weight` (IQ1_M, 18,874,368
  elements) and of `blk.3.ffn_down_exps.weight` (IQ2_XXS) both give `max |strata - ggml| = 0`. So the slot byte
  stride (`total_bytes / n_experts`, block-aligned) and the IQ decoders are not the divergence. The check is
  `/tmp/oracle.cpp` on this machine: link `build-m3/libstrata_quant.a` with `-lggml`, use
  `ggml_get_type_traits(type)->to_float`, and read the byte offset as `tensor.data.offset + tensor.data_offset`
  - only for 3-D stacks, since a 2-D tensor has no expert stride.
- gguf-py cannot dequantise IQ1_M for the numpy path, so the numpy cross-check stops at the dense layers; use
  ggml as the second decoder for quantised tensors instead.

Live hypothesis, next thing to read: the MSA attention path itself. `build_attn_msa_fa(Qcur, kfa, vfa, mfa, Gp,
kq_scale, il)` is handed `Gp = n_head / n_head_kv` (16), and the no-flash-attn fallback branch of
`minimax-m3.cpp` (the `zero`/`pos_slot`/`ggml_scale` block around L460-520) is what actually ran in the reference
run - flash attention was off. Read that branch op by op against `forward.hpp` before writing any kernel: if the
sparse path pools or scales by `Gp`, or masks beyond the selected blocks differently, every sparse layer inherits
it and the dense layers stay correct - exactly the symptom.
