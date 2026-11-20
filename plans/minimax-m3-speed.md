# MiniMax-M3 speed: status and the honest ceiling

Hardware: 2× RTX 5060 Ti (16 GB, PCIe Gen3 x8, ~6 GB/s pinned H2D measured) + 1× RTX 3060
(12 GB, Gen3 x16, ~12 GB/s), 256 GB RAM. Model: 112 GB AD-IQ2_XXS (~90% MoE experts:
57 sparse layers × 128 experts, top-4 + 1 shared; 3 leading dense layers).

## Measured progression (all correct token 200058, Moon prompt, one box)
**Final: ~2.2 tok/s sustained (50-token generation) - a ~1100x speedup over the 0.002 tok/s
start, at ~80% of the estimated 2.5-3 tok/s hardware ceiling.**

| Stage | ms/step | tok/s | What changed |
|---|---|---|---|
| All-F32 original | 492,000 | 0.002 | CPU dequant + pageable F32 uploads |
| `--quant` batched forward | 2,200 (est. per token) | ~0.45 | iq_mmvq_f32 (quant weights × float activation), experts streamed from mmap |
| Decode engine (resident weights + persistent KV + 1-token steps) | ~1,530 | 0.65 | no more prefix recompute / weight re-upload |
| + pinned-RAM model staging (112 GB page-locked once) | ~860 | 1.16 | expert H2D 2.93 → 6.02 GB/s (2×), truly async |
| + 3-GPU expert parallelism (expert e on GPU e%3, phase A/B/C) | ~610 | 1.64 | per-layer expert H2D split across 3 links concurrently |
| + per-GPU expert VRAM LRU on the extra GPUs (256 slots each) | ~516 @ step 7, falling | **~1.9 warming** | hits skip PCIe entirely; routed section decays 366→159 ms over 7 steps |
| + GPU0's expert LRU inside the multi-GPU path | ~455 | **~2.2** | GPU0 LRU 67% hit; its 1/3 expert-ID space fits 128 slots |
| + 20 MB slots (was 32 MB), 384 extra-GPU slots default | ~445 | **~2.2** | ~60% more cached instances in the same VRAM; GPU1/GPU2 ~86% hit |

## Where the step time goes (3-GPU + all-LRU, step P~55, M3_TPROFILE)
- attn ~180-195 ms + router ~127-138 ms ≈ 310-330 ms combined: this is **two full GPU-stream
  drains per sparse layer** - the MSA indexer D2Hs the key cache + host-scores blocks +
  `cudaStreamSynchronize`, then the router D2Hs scores + host top-k + `cudaStreamSynchronize`.
  Each drain serializes the GPU's attention compute (q/k/v/o GEMV + attention + indexer), paid
  57x/token. The "attn" wall time is CPU launch + the indexer drain wait; the "router" wall time
  is the router drain wait + host top-k. Both grow slowly with context (MSA reads more blocks).
- routed experts ~110-123 ms (cache-aware: GPU0 LRU 67% hit, GPU1/GPU2 LRU ~86% hit; the rest is
  miss-streaming at 6 GB/s + the relay/sync overhead, inherent to host-mediated dispatch)
- shared expert ~1.7 ms, output head ~12 ms
- Cache hit rates are high because the greedy base-model generation falls into a repetition loop
  whose activations keep the router selecting a recurring expert subset (good locality).

## The honest ceiling (corrects the earlier "~10 tok/s")
Per token ~3 GB of routed-expert weights must be read. With 3 concurrent PCIe links + VRAM
caches of the hot experts, the floor is roughly: attention compute (~170 ms, serial at batch 1)
+ router sync (~70 ms) + remaining expert streaming/GEMV (~70 ms when warm) ≈ 0.35-0.45 s/token
≈ **2.5-3 tok/s**. We are at ~2.2 sustained. 10 tok/s is not physically reachable on this box
(would need ~30 GB/s effective expert-weight bandwidth; layers are sequential at batch 1, so
attention stays serial).

## Key findings (measured, not estimated)
- The routed-expert time is **H2D streaming**, not GEMV compute: iq_mmvq_f32 runs at 111-128
  Gelem/s (all 7 model types), so all expert GEMVs cost ~0.12 s/token; the ~3 GB/token of
  expert bytes at 2.93 GB/s (pageable) / 6.02 GB/s (pinned) dominate.
- Cross-token expert reuse is **only ~10%** per token-pair for this model (measured hit rate
  on a 128-slot cache). The VRAM LRUs still win because they turn recurring experts into free
  reads; a plain LRU thrashes (a token's 228-instance working set exceeds any small cache),
  hence the frequency-aware policy (auto-pin at 8 uses, capped at half the arena, LFU refresh
  after the prompt pass).
- A token's working set (228 unique instances ≈ 3 GB) does not fit in one GPU's spare VRAM
  next to the 9.1 GB resident weights — the single-GPU VRAM wall. Multi-GPU is the only way
  to have both resident weights and a big expert cache.
- Consumer GPUs: no P2P, so inter-GPU data moves via pinned host relay buffers (24 KB each,
  negligible).

## Done
- [x] P0 all-F32 correct baseline (492 s).
- [x] P1 iq_mmvq_f32 kernel (quant weights × float activation), all 7 types, oracle-verified
      <2e-6 vs CPU dequant; correct 200058 at 31× the F32 speed.
- [x] P1 decode engine (resident weights + persistent KV + single-token steps).
- [x] P1 pinned-RAM model staging (M3_NOSTAGE=1 to disable).
- [x] P1 frequency-aware expert cache (pin + LFU refresh) on GPU0 (--vexperts N).
- [x] P2 3-GPU expert-parallel decode (--gpus N, default 3).
- [x] P2 per-GPU expert VRAM LRU on the extra GPUs (--expslots N, default 256).

## Remaining (diminishing returns; the ceiling is ~2.5-3 tok/s)
Both remaining levers are multi-hour efforts for +0.3-0.5 tok/s:
- [ ] **Tensor-parallel the q/o GEMVs** across the 3 GPUs (split output rows; relay ~100 KB/layer
      back). Cuts the serialized attention GPU compute ~2/3 for those two GEMVs -> ~40-80 ms/token
      -> ~2.5-2.7 tok/s. Moderate refactor (weights on 3 GPUs, device-aware GEMV dispatch).
- [ ] **Fused GPU-side MoE dispatch + GPU-side MSA block selection** (no host round trip). Would
      eliminate the two per-layer stream drains (the bulk of the 310 ms attn+router) but is a large,
      high-risk kernel effort (dynamic expert dispatch reading quantised weights in VRAM/pinned RAM).
- [ ] Fix the pre-existing `free(): double free` at process exit after a serve run
      (cosmetic; all tokens are produced before it).

## Flags
`--quant` (required for the fast path) `--gpus N` (default 3) `--expslots N` (extra-GPU LRU
slots, default 384 ≈ 7.5 GB arena each at 20 MB/slot) `--vexperts N` (GPU0's expert LRU slots,
20 MB each) `--ctx N`
(M3_NOSTAGE=1 env: skip the one-time pinned-RAM staging) `M3_TPROFILE=1` env: per-section
step timing.
