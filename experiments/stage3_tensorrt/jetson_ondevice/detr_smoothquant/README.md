# DETR on-device SmoothQuant — does the activation-granularity lever recover the INT8 collapse?

The DETR accuracy axis (`../detr_accuracy/`, commit `58ba518`) built the first
**accuracy-valid** explicit-symmetric INT8 DETR engine on Orin and found it **reproduces
stage2's mAP collapse** (FP32 **0.4237** → INT8 **0.2383**, −43.8%; mAP_s −84.6%). Its
verdict: the case-C fix is a *toolchain* unlock, not an *accuracy* rescue — and it pointed
at the real lever, **activation quantization granularity** (SmoothQuant,
`study_guide/04_transformer_quantization.md` §4.4). stage2 §4.4 recovered **59.9%** of a
per-tensor INT8 collapse with SmoothQuant — but **only on the torch fake-quant path
(Design X)**, which never produced an ONNX and so was **never measured on-device**.

This run closes that gap: apply SmoothQuant **at the ONNX-graph level** to the same
board / same 1000 images / same symmetric QDQ recipe, and measure whether the on-device
collapse recovers.

## Headline

**On-device SmoothQuant recovers only ~9% of the DETR INT8 collapse — an order of
magnitude below stage2 §4.4's 59.9%.** The root cause is **op coverage**, not SmoothQuant:
the only buildable on-device INT8 engine leaves attention MatMul + LayerNorm + Softmax in
FP16 (forced by the case-C parser + quantized-constant builder walls), so INT8 is
**Gemm-only** and smoothing Gemm inputs is not the dominant residual-error lever there.
stage2's torch path had **every linear** (incl. attention Q/K/V/out) in per-tensor INT8,
where SmoothQuant bites hard. SmoothQuant is still **directionally correct** (α=1.0 > α=0.5,
matching stage2's DETR-best α ordering; small-object mAP_s partially recovers) and
**nearly latency-neutral** (+0.34 ms vs sym). → stage2 §4.4's 59.9% is not wrong, it is
**path/op-coverage dependent**.

| engine | latency (ms) | engine (MiB) | mAP | mAP_s | Δ vs sym | gap recovery |
|---|---|---|---|---|---|---|
| iGPU FP32 (ceiling) | 25.8496 | 160.12 | 0.4237 | 0.2179 | +0.1854 | 100% |
| iGPU INT8 explicit-sym (floor) | 11.0020 | 44.69 | 0.2383 | 0.0336 | — | 0% (the collapse) |
| **+ SmoothQuant α=1.0** | **11.3433** | 44.87 | **0.2553** | 0.0449 | **+0.0170** | **9.2%** |
| + SmoothQuant α=0.5 | 11.4084 | 44.91 | 0.2544 | 0.0438 | +0.0161 | 8.7% |
| _(stage2 §4.4 ref · same lever, torch)_ | — | — | _+0.0544_ | — | — | **59.9%** |

gap = FP32 − sym = **0.1854**. mAP_s recovery: α=1.0 **6.1%**, α=0.5 5.5%.
FP32/sym rows are re-evaluated from the committed `../detr_accuracy/results/*.npz` with the
**same** postprocess and reproduce 0.4237 / 0.2383 exactly (no pipeline drift = the
cross-check). The SmoothQuant rows differ from the committed `detr_gpu_int8_sym.plan` by
**one thing only**: the ONNX-level SmoothQuant activation migration.

## Method

- Board: **NVIDIA Jetson AGX Orin Developer Kit (64GB)** · JetPack 6.2.1 (L4T R36.4.3) · CUDA 12.6 · **TensorRT 10.3.0** (`trtexec` banner `v100300`) · Ampere iGPU · `nvpmodel MAXN`
- Runner: reuses `../detr_accuracy/scripts/orin_detr_map.py::run_engine` (batch1, `execute_async_v3`, the `.copy()` zero-copy-aliasing guard); board dumps **raw logits/boxes** (`.npz`), host computes pycocotools mAP.
- Preprocess: **force-resize 800×1066** (matches the FIXED `detr_sim.onnx` shape) + ImageNet norm, byte-identical between host calibration and board eval (`../detr_accuracy/scripts/detr_prep.py`).
- Eval = COCO val2017 **head 1000**; SmoothQuant activation statistics = **tail 100** (disjoint).

### SmoothQuant math (ONNX-level, manual)

All **95 DETR Gemms** are `transA=0, transB=0` (asserted): `Y = A·B`, weight `B=[K,N]`,
contraction axis `K = A.shape[-1] = B.shape[0]` (weight **rows**). Per-input-channel(K)
migration scale:

```
s_k = a_k^α / w_k^(1-α)          # a_k = per-channel(K) activation absmax (calib 100 imgs)
                                 # w_k = per-row(axis0) weight absmax
A'[:,k] = A[:,k] / s_k           # inject Mul(A, 1/s) before the Gemm (broadcast last axis)
B'[k,:] = B[k,:] · s_k           # scale weight row k  (W * s[:,None])
A'·B' = A·B                      # EXACT for any s>0 — outliers migrate activation→weight
```

The benefit is entirely on the activation side: per-channel activation outliers move into
the weights, so the **per-tensor activation quantization** after migration is tighter.

- **modelopt.onnx is not importable** here (`onnxslim` missing) → forced the manual ONNX
  path. This is also *why* stage2 §4.4 never went on-device (it lived on torch fake-quant).
- **FP32 exactness gate must run on `CPUExecutionProvider`.** On CUDA EP the gate spuriously
  fired at `max_abs≈0.65` — Ampere **TF32** matmul (10-bit mantissa) is sensitive to
  SmoothQuant's operand rescaling (a precision artifact, *not* a wiring bug; pure-numpy
  SmoothQuant math is exact to 1.4e-14). On CPU EP: **1.748e-4 (α=1.0) / 1.450e-4 (α=0.5)**,
  both `< 1e-2` → PASS.

### Symmetric requantization (identical to `58ba518`, clears 3 walls)

```python
quantize_static(SMOOTH, DST, quant_format=QDQ,
    activation_type=QInt8, weight_type=QInt8, per_channel=True,
    op_types_to_quantize=["Conv", "Gemm"],                 # attention MatMul/LayerNorm/Softmax stay FP16 (builder wall)
    nodes_to_exclude=["/model/backbone/model/conv1/Conv"], # case D (stem conv)
    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,  # zp=0 → clears case C (shiftIsAllZeros)
                   "QuantizeBias": False})                  # no INT32 bias DQ → clears case B
# VERIFY: zp_total=732  zp_nonzero=0  int32_bias_DQ=0  SQMul_kept=95
# The QDQ graph minus the SmoothQuant Muls is byte-identical to the sym engine (43,612,561 B).
```

SmoothQuant activation absmax compression (how far outliers migrated):

| α | max | at | median | ≥2× Gemms | FP32 gate |
|---|---|---|---|---|---|
| 1.0 (a10) | **46.699×** | Gemm_4 | 7.485× | 94/95 | 1.748e-4 |
| 0.5 (a05) | 19.265× | Gemm_4 | 5.51× | 95/95 | 1.450e-4 |

## Reproduce

```bash
# host (emb-ai venv) — 1) ONNX SmoothQuant + symmetric requant + FP32 exactness gate (CPU EP)
python scripts/detr_sq_export.py --alpha 1.0 --tag a10
python scripts/detr_sq_export.py --alpha 0.5 --tag a05
#   → _workspace/stage3_jetson_detr_sq/detr_int8_sq_{a10,a05}.onnx  + results/sq_export_meta_{a10,a05}.json

# board (~/orin_bench) — 2) build engines with the SAME trtexec flags as 58ba518, then dump raw
#   trtexec --onnx=onnx/detr_int8_sq_a10.onnx --int8 --fp16 --saveEngine=engines/detr_gpu_int8_sq_a10.plan \
#           --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100   (→ raw/detr_sq_build_a10.log)
python scripts/orin_detr_sq_map.py     #   → ~/orin_bench/detr_smoothquant/detr_sq_{a10,a05}_raw.npz

# host — 3) pycocotools mAP + recovery (re-evaluates committed fp32/sym npz as the cross-check)
python scripts/analyze_detr_sq_map.py  #   → results/detr_sq_map_summary.json
python scripts/build_sq_summary.py     #   → results/detr_sq_summary.json  (SSOT)
```

## Files

- `scripts/detr_sq_export.py` — ONNX-level SmoothQuant (95 Gemms) + symmetric requant + FP32 exactness gate (CPU EP). Args `--alpha`, `--tag`.
- `scripts/orin_detr_sq_map.py` — board dumper; reuses `orin_detr_map.run_engine`.
- `scripts/analyze_detr_sq_map.py` — host pycocotools mAP + recovery-of-gap; re-evaluates committed fp32/sym npz (cross-check → 0.4237/0.2383).
- `scripts/build_sq_summary.py` — SSOT builder (`detr_sq_summary.json`); recomputes all ratios.
- `results/detr_sq_{a10,a05}_raw.npz` — per-engine raw logits/boxes/img_ids (board dump).
- `results/detr_sq_map_summary.json` — mAP + recovery + divergence.
- `results/sq_export_meta_{a10,a05}.json` — SmoothQuant compression stats + verify + exactness.
- `results/orin_detr_sq_meta.json` — board/TRT/engine metadata.
- `results/detr_sq_summary.json` — **SSOT** (report + memory read from here).
- `raw/detr_sq_build_{a10,a05}.log` — trtexec build+profile (GPU Compute median 11.3433 / 11.4084 ms).
- `raw/detr_sq_dump.log` — board dump stdout (1000/1000 both engines).
- `raw/detr_sq_engine_sizes.txt` — board `ls -la` of the two engines (47052636 / 47092836 B) + `trtexec v100300`.

## Caveats

- Absolute mAP is **not comparable** to stage2 §4.4 (fixed-800×1066 force-resize vs stage2 dynamic shape; TRT vs ORT torch fake-quant; 1000 vs 5000 images). Only the **relative recovery-of-gap** is the result — and it (9.2%) is what refines stage2's 59.9% into "path/op-coverage dependent".
- 1000-image head subset (stage1 함정 0: subset mAP is itself inflated); same-bundle relative deltas only.
- SmoothQuant is FP32-exact by construction (`s` cancels). The on-device benefit comes purely from tighter per-tensor activation quant after outlier migration — verified the engines are real INT8 (FP32 logits_corr ~0.9826) and genuinely differ from the sym engine (logits_absmax ~16.4 vs sym), ruling out a silent FP32/sym clone.
- Jetson is NVIDIA edge silicon, **not** one of the three automotive vendors (TI / Qualcomm / Renesas).
