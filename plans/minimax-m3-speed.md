# MiniMax-M3 speed: status and the honest ceiling

Hardware: 2× RTX 5060 Ti (16 GB, PCIe Gen3 x8, ~6 GB/s pinned H2D measured) + 1× RTX 3060
(12 GB, Gen3 x16, ~12 GB/s), 256 GB RAM. Model: 112 GB AD-IQ2_XXS (~90% MoE experts:
57 sparse layers × 128 experts, top-4 + 1 shared; 3 leading dense layers).

## Measured progression (all correct token 200058, Moon prompt, one box)

| Stage | ms/step | tok/s | What changed |
|---|---|---|---|
| All-F32 original | 492,000 | 0.002 | CPU dequant + pageable F32 uploads |
| `--quant` batched forward | 2,200 (est. per token) | ~0.45 | iq_mmvq_f32 (quant weights × float activation), experts streamed from mmap |
| Decode engine (resident weights + persistent KV + 1-token steps) | ~1,530 | 0.65 | no more prefix recompute / weight re-upload |
| + pinned-RAM model staging (112 GB page-locked once) | ~860 | 1.16 | expert H2D 2.93 → 6.02 GB/s (2×), truly async |
| + 3-GPU expert parallelism (expert e on GPU e%3, phase A/B/C) | ~610 | 1.64 | per-layer expert H2D split across 3 links concurrently |
| + per-GPU expert VRAM LRU on the extra GPUs (256 slots each) | ~516 @ step 7, falling | **~1.9 warming** | hits skip PCIe entirely; routed section decays 366→159 ms over 7 steps |
| Sustained, 50-token generation (P up to 55) | ~540 | **~1.85** | extra-GPU LRUs at steady hit rate (routed ~130 ms); attn/router grow with context |

## Where the step time goes (3-GPU + LRU, step P=7, M3_TPROFILE)
- attn ~269 ms (includes ~100 ms of queue-flush artifact; true attention compute ~171 ms)
- router ~74 ms (D2H scores + host top-k + stream sync, 1.2 ms/layer)
- routed experts ~159 ms and falling (extra-GPU LRU warming; cold = 366 ms)
- shared expert ~1.7 ms, output head ~12 ms

## The honest ceiling (corrects the earlier "~10 tok/s")
Per token ~3 GB of routed-expert weights must be read. With 3 concurrent PCIe links + VRAM
caches of the hot experts, the floor is roughly: attention compute (~170 ms, serial at batch 1)
+ router sync (~70 ms) + remaining expert streaming/GEMV (~70 ms when warm) ≈ 0.35-0.45 s/token
≈ **2.5-3 tok/s**. We are at ~1.9 and rising with warmup. 10 tok/s is not physically reachable
on this box (would need ~30 GB/s effective expert-weight bandwidth; layers are sequential at
batch 1, so attention stays serial).

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
- [x] Warm-state measurement over a 50-token generation (P up to 55): **sustained ~1.85 tok/s**
      (step ~540 ms). The routed section plateaus ~125-135 ms once the extra-GPU LRUs reach
      steady hit rate; attn grows slowly to ~270-290 ms and router to ~125-135 ms as context
      lengthens (MSA reads more selected blocks; the router's stream-sync flushes more queued
      expert work). Both are context-length effects, not regressions.
- [ ] Faster attention path (the last ~170 ms of serial compute; multi-hour kernel work).
- [ ] Router without the host sync (port mm::route's top-k to a GPU kernel; saves ~70 ms).
- [ ] Fix the pre-existing `free(): double free` at process exit after a serve run
      (cosmetic; all tokens are produced before it).

## Flags
`--quant` (required for the fast path) `--gpus N` (default 3) `--expslots N` (extra-GPU LRU
slots, default 256 ≈ 8 GB arena each) `--vexperts N` (GPU0's expert LRU slots) `--ctx N`
(M3_NOSTAGE=1 env: skip the one-time pinned-RAM staging) `M3_TPROFILE=1` env: per-section
step timing.
