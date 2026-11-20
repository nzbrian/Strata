# MiniMax-M3 speed: status and the honest ceiling

Hardware: 2× RTX 5060 Ti (16 GB, PCIe Gen3 **x8** ≈ 6 GB/s usable each) + 1× RTX 3060 (12 GB,
Gen3 **x16** ≈ 12 GB/s usable), 256 GB RAM. Model: 112 GB AD-IQ2_XXS (≈90% MoE experts:
57 sparse layers × 128 experts, top-4 + 1 shared; 3 leading dense layers).

## Where we are (measured on one 5060 Ti)

| Path | Result |
|---|---|
| All-F32 (original) | 492 s per 7-token forward, correct token 200058 |
| `--quant` batched forward (quant-weight × float-activation GEMV) | **15.8 s** per 7-token forward, **correct 200058** (31× faster) |
| **Decode engine** (resident weights + persistent KV + 1-token steps) | **~2.06 s/token ≈ 0.49 tok/s**, correct token sequence |

The `--quant` path and the decode engine both reproduce the reference token 200058 and match the
batched forward's logits. This is a working, correct, measured M3 CUDA decode path.

## The honest ceiling (corrects the earlier "~10 tok/s")

Per decode token the model must read **~3 GB of routed-expert weights** (228 expert uses × ~13 MB
quantised bytes) that do not fit in VRAM (experts total ~100 GB; 44 GB VRAM across all GPUs).

- **Layers are strictly sequential**, so only ONE GPU's PCIe link streams experts at a time. The
  3 GPUs give 3× *VRAM* (a bigger expert cache), NOT 3× *bandwidth*.
- Single GPU: ~3 GB/token over ~6 GB/s ≈ 0.5 s/token of pure streaming + GEMV compute →
  **~0.5–1 tok/s** is the single-GPU ceiling (we measure 0.49).
- All 3 GPUs (weights + KV + expert LRU spread across 44 GB): a thrash-free expert LRU caches a
  large fraction of the ~3 GB/token, cutting PCIe traffic → **~2–4 tok/s** is the realistic
  ceiling.
- **10 tok/s is not physically reachable** on this box for this model. It would need ~30 GB/s of
  effective expert-weight bandwidth, which the single-active-PCIe-link + VRAM-capacity limits do
  not provide. (My earlier "10 tok/s ceiling" estimate wrongly assumed 32 GB/s aggregate PCIe.)

## The single-GPU VRAM wall (why vexp didn't help on one GPU)

Resident projection weights ≈ 9.1 GB. A useful expert LRU needs ≥ ~8 GB (a token's working set is
228 distinct (layer,expert) instances; a 128-slot LRU evicts itself mid-token and thrashes —
measured SLOWER than no cache). 9.1 + 8 + KV > 16 GB, so on one 5060 Ti you must choose between
resident weights and a useful expert cache. Multi-GPU is the only way to have both.

## Done
- [x] P0: all-F32 correct baseline (492 s).
- [x] P1: quant-weight × float-activation GEMV kernel (`iq_mmvq_f32`), all 7 model types, oracle-
      verified against CPU dequant to <2e-6; correct 200058 at 15.8 s (31×).
- [x] P1: VRAM expert LRU (`--vexperts N`, pinned-staging misses) — correct, but thrashes below
      the per-token working set; useful only with the multi-GPU VRAM headroom.
- [x] P1: decode engine (resident weights + persistent KV + single-token steps) — correct,
      measured 0.49 tok/s single-GPU.

## Remaining (only if the user wants the multi-GPU push)
- [ ] **S2: 3-GPU layer split.** Split the 60 layers across the 3 GPUs (each holds its layers'
      resident weights + its own expert LRU + KV). Sequential layer dependency means one GPU is
      active at a time, but the aggregate 44 GB VRAM holds a thrash-free expert LRU → the 2–4
      tok/s ceiling. This is a multi-hour rearchitecture (multi-device CUDA context, per-GPU
      placement, cross-GPU residual handoff).
- [ ] F16 activations / a faster expert GEMV to cut the compute half of the ~2 s/token.
- [ ] Fix the pre-existing `free(): double free` at process exit after a serve run (cosmetic;
      does not affect the generated output).

## Notes
- Correctness oracle: the all-F32 path and the batched `--quant` forward both give top-1 200058
  for the Moon prompt; the decode engine matches.
- The pre-existing `free(): double free detected in tcache` fires at process exit after any serve
  run (present before the decode engine); all tokens are produced before it.
