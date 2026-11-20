# MiniMax-M3 speed: 0.002 → ~10 tok/s on 3× 5060 Ti + 1× 3060, 256 GB RAM

Target ~10 tok/s decode. 20 tok/s is out of reach (model 112 GB > 44 GB VRAM; routed
experts not in VRAM must cross PCIe Gen3 x8/x8/x16 ≈ 32 GB/s aggregate → ~10 tok/s ceiling).

## Hardware
- GPU0/1: RTX 5060 Ti, 16 GB, PCIe Gen3 x8 (≈8 GB/s each)
- GPU2: RTX 3060, 12 GB, PCIe Gen3 x16 (≈16 GB/s)
- 256 GB RAM. Model 112.2 GB (≈90% MoE experts, 128/layer top-4 + 1 shared, 57 sparse layers).
- Per-token weight reads (quantised): ~5 GB routed experts + ~1.5 GB attention + ~0.8 GB embed/head.

## Ceiling math
Attention+embed+KV in VRAM (~4 GB). ~39 GB VRAM left for a hot-expert LRU (~39% of experts).
Routed experts not in VRAM ≈ 0.61 × 5 GB ≈ 3.0 GB/token over 32 GB/s ≈ 95 ms → ~10.5 tok/s.

## Phases (measure each)
- **P0 baseline** (done): all-F32 single GPU = 492 s/token (0.002 tok/s). `--quant` (Q8_1 act)
  baseline: <pending>.
- **P1 correct+fast single-GPU quantised path** (~1.5–2 tok/s):
  1. Quantised-weight × **BF16-activation** GEMV (iq_mmvq is Q8_1-only; Q8_1 act compounds to a
     wrong token over 60 layers). [kernel]
  2. Weights resident: attention+embed in VRAM, experts in **pinned** RAM; async double-buffered
     H2D (no per-token re-upload from the mmap). [engine]
  3. Verify: 4-layer logits match the all-F32 reference; full model → 200058. Measure.
- **P2 3-GPU** (~10 tok/s):
  4. Layer split across 3 GPUs; each streams its experts over its own link.
  5. Expert VRAM LRU (per-GPU) for the hot experts.
  6. Verify 200058; measure toward 10 tok/s.

## Notes
- Correctness oracle: all-F32 path gives top-1 200058 (the reference). 4-layer logit diff is the
  fast check; full run is the milestone check.
- Keep the all-F32 path intact (default) as the correctness reference.
