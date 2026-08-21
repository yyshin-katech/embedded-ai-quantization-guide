# DETR accuracy-valid INT8 on-device — closing the "정확도 미측정" caveat (case-C bypass)

The DETR latency run (`../detr/`, commit `9ef2a58`) proved the **transformer** story on
Orin — INT8 arithmetic gain returns on a compute-bound transformer (×2.742), DLA
fragments a transformer (404 GPU-fallback / 16 islands), and **explicit-QDQ INT8 would
not even parse** on `trtexec` (stage3 **case C**: `Assertion failed:
shiftIsAllZeros(zeroPoint)`). So that run reported latency only; it explicitly did **not**
measure DETR INT8 accuracy on-device. This follow-on closes that gap the way the
ResNet50 axis (`../accuracy/`) did — but first it has to get an accuracy-valid INT8
engine to exist at all.

**Two walls, two fixes** (see `../parser_constraints.md` for the stage3 case taxonomy):

1. **case C (parser, `zp≠0`) + case B (INT32 bias DQ)** — the stage2 `detr_int8.onnx` used
   ORT defaults → zp≠0 in 1085/1485 Q/DQ + 149 INT32 bias DQ → parser dies at node 0. Fix
   = the **same symmetric recipe that let ResNet50 build** (`../t02_latency_3point.py`):
   `ActivationSymmetric+WeightSymmetric` (zp=0 everywhere → clears C), `QuantizeBias=False`
   (no INT32 bias DQ → clears B), exclude backbone `conv1` (case D stem 3ch7×7).
2. **NEW wall — quantized-constant-in-self-attention (builder, transformer-specific).**
   With the symmetric fix but ORT's *default* op coverage (it quantizes **every** op:
   Reshape/Transpose/Softmax/Mul/MatMul/Where/Constant…), the parse now passes but the
   **builder** rejects it:
   ```
   Error Code 2: [qdqGraphOptimizer.cpp::matchQuantizedConstantPluginOrDQ::4055]
   .../self_attn/Constant_3_output_0_quantized: Quantized constant is only allowed
   before DQ or PLUGIN_V2 or kPLUGIN_V3 node
   ```
   Fix = `op_types_to_quantize=["Conv","Gemm"]` (quantize only the weight-bearing ops;
   leave attention matmuls + LayerNorm + Softmax in FP16). This is exactly **stage2 §4.5**'s
   finding — op-selection mixed barely moves mAP; the lever is activation granularity, not
   op set.

Result: `detr_int8_sym.onnx` (43.4 MB, `zp_nonzero=0`, `int32_bias_DQ=0`) → the
`gpu_int8_explicit` config that was **build-failed in 9ef2a58 now builds** (44.69 MiB
engine, 11.002 ms). The case-C fix is a **toolchain unlock**. Whether it is also an
**accuracy** unlock is the question this run answers.

- Board: **NVIDIA Jetson AGX Orin Developer Kit (64GB)** · JetPack 6.2.1 (L4T R36.4.3) · CUDA 12.6 · **TensorRT 10.3.0** · Ampere iGPU · `nvpmodel MAXN`
- Runner: `tensorrt` Python API + `cuda-python` 12.9.7 (`from cuda.bindings import runtime`) — batch1, `execute_async_v3`; board dumps **raw logits/boxes** (`.npz`), host computes mAP (board has no pycocotools)
- Preprocess: **force-resize 800×1066** (matches the FIXED `detr_sim.onnx` shape) + ImageNet norm, byte-identical between host calibration and board eval (`scripts/detr_prep.py`)
- Eval set: **COCO val2017 head 1000** (sorted img_ids); calibration = **tail 100** (disjoint)
- Metric: **pycocotools bbox mAP** on host; postprocess = the stage2 `s2_07` formula (softmax → drop no-object → cxcywh×(W,H) → xywh), scaled by each image's **original** W,H (DETR boxes are resolution-normalised, so the input resize does not enter the box math)

## How to reproduce

```bash
# host (emb-ai venv): symmetric re-export (bypass case C/B + attention builder wall)
~/emb-ai/bin/python scripts/detr_sym_export.py         # -> _workspace/.../detr_int8_sym.onnx (VERIFY zp_nonzero=0, int32_bias_DQ=0)
~/emb-ai/bin/python scripts/prep_coco_sub.py 1000      # -> coco_sub.tar (head 1000 + manifest.json)

# board (~/orin_bench): build the accuracy-valid INT8 engine, then dump raw outputs
scp detr_int8_sym.onnx jetson:~/orin_bench/onnx/ ; scp coco_sub.tar jetson:~/orin_bench/
/usr/src/tensorrt/bin/trtexec --onnx=onnx/detr_int8_sym.onnx --int8 --fp16 \
    --saveEngine=engines/detr_gpu_int8_sym.plan --warmUp=2000 --duration=10 --iterations=200 --avgRuns=100
python3 orin_detr_map.py                               # dumps detr_{fp32,fp16,int8_sym,int8_implicit}_raw.npz

# host: pull *.npz into results/, compute mAP + assemble SSOT
scp 'jetson:~/orin_bench/detr_accuracy/*.npz' results/
~/emb-ai/bin/python scripts/analyze_detr_map.py        # -> results/detr_map_summary.json
~/emb-ai/bin/python scripts/build_summary.py           # -> results/detr_accuracy_summary.json (SSOT)
```

## Results (SSOT = `results/detr_accuracy_summary.json`)

Latency/engine bytes for FP32/FP16/implicit are the **committed 9ef2a58 SSOT**
(`../detr/results/detr_summary.json`, identical flags); `int8_sym` latency/bytes are this
session's build. mAP is pycocotools bbox over the head-1000 subset at fixed 800×1066.

| engine | latency (median) | vs FP32 | engine | **mAP** | mAP_s | vs FP32 | accuracy |
|---|---:|---:|---:|---:|---:|---:|---|
| iGPU FP32 | 25.8496 ms | ×1.00 | 160.12 MiB | **0.4237** | 0.2179 | — | valid |
| iGPU FP16 | 13.2754 ms | ×1.947 | 81.43 MiB | **0.4243** | 0.2207 | +0.0006 (lossless) | valid |
| **iGPU INT8 explicit-sym** | **11.002 ms** | **×2.350** | **44.69 MiB** | **0.2383** | **0.0336** | **−0.1854 (−43.8%)** | **valid** |
| iGPU INT8 implicit `--int8` | 9.42871 ms | ×2.742 | 58.76 MiB | 0.4073 | 0.1892 | −0.0164 (−3.9%) | **NOT claimed** |

### Headline — the case-C fix is a *toolchain* unlock, not an *accuracy* rescue
The symmetric re-export makes explicit-QDQ INT8 **buildable** (it was build-failed in
9ef2a58). But the accuracy-valid engine **reproduces the stage2 DETR collapse**:
FP32 0.4237 → INT8 **0.2383 (−43.8%)**, small-object mAP_s 0.2179 → **0.0336 (−84.6%)**.
Compare stage2 (ORT-QDQ, dynamic shape, CUDA EP, 5000 img): 0.4207 → 0.2402 (**−42.9%**),
mAP_s **−77%**. Two different paths (TRT/fixed-shape/1000 vs ORT/dynamic/5000) land on the
**same collapse** → cross-validated. **The lever is not the sym/case-C fix — it is
activation quantization granularity (SmoothQuant, stage2 §4.4 recovered 59.9% of the
per-tensor gap).**

### Why implicit `--int8` (0.4073) is NOT a better INT8 — its accuracy is nominal
`trtexec --int8` with **no calibration cache** prints (see
`raw/implicit_no_calibrator_warning.txt`):
```
[W] [TRT] Calibrator is not being used. Users must provide dynamic range for all
tensors that are not Int32 or Bool.
```
No calibration → TRT has **no data-derived activation ranges**. It still runs INT8 kernels
— the implicit engine is **faster than FP16** (9.43 < 13.28 ms) and **smaller** (58.76 <
81.43 MiB), so INT8 kernels *are* active; it is **not** a pure FP16 fallback — but on
**uncontrolled (non-data-derived) ranges**. So its INT8 label is nominal and **accuracy is
not claimed**: here it happens to land near FP16 (0.4073, −3.9%), but the companion from
the ResNet50 axis — DLA implicit `--int8` — collapsed to **0.017** on the same uncontrolled
path. Either way, only an **explicit, calibrated** QDQ engine gives an accuracy you can
stand behind.

### Numerical divergence vs FP32 (secondary, weak proxy)
`div` in the SSOT: FP16 logits_corr 0.99991 / boxes_corr 0.99944 (near-identical);
sym 0.98236 / 0.94532; implicit 0.98038 / 0.94062. Note sym and implicit have **nearly
identical** global corr despite a 0.17 mAP gap — the Pearson corr over all 100×92 logits
is dominated by the no-object logit dimension, so it is a **weak** proxy. **pycocotools
mAP is the authoritative metric.**

## Caveats (unchanged from the axis)
- Absolute mAP is **not** comparable to stage2 (fixed-800×1066 force-resize distorts aspect
  ratio vs stage2's dynamic shape; TRT vs ORT; 1000 vs 5000 images). Only the **relative**
  FP32→INT8 collapse is the result — and that relative collapse cross-validates stage2.
- 1000-image head subset (stage1 함정 0 top-1 inflation applies to mAP too); same-bundle
  relative deltas only.
- `int8_implicit` accuracy (0.4073) is **not claimed** (uncalibrated → uncontrolled auto-ranges, not a pure FP16 fallback).
- Jetson is NVIDIA edge silicon, not one of the three automotive vendors (TI/Qualcomm/Renesas).

## Files
- `scripts/detr_prep.py` — shared preprocess (host calib + board eval, byte-identical)
- `scripts/detr_sym_export.py` — symmetric QDQ re-export (bypass case C/B + attention builder wall); VERIFY asserts zp_nonzero=0, int32_bias_DQ=0
- `scripts/prep_coco_sub.py` — stage head-1000 COCO subset + manifest, tar for scp
- `scripts/orin_detr_map.py` — on-board raw-output dumper (4 engines → npz)
- `scripts/analyze_detr_map.py` — host pycocotools mAP + divergence
- `scripts/build_summary.py` — assemble `results/detr_accuracy_summary.json` (SSOT)
- `results/` — `detr_*_raw.npz` (per-engine raw outputs), `detr_map_summary.json`, `detr_accuracy_summary.json`, `orin_detr_meta.json`
- `raw/` — `int8_sym_build_PASSED.txt` (trtexec build log), `implicit_no_calibrator_warning.txt` (mechanism), `onboard_dump_run.txt`
