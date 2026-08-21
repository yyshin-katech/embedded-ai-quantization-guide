# On-device accuracy — Jetson AGX Orin (does INT8 accuracy hold on the silicon?)

The solo sweep (commit `00fd97d`) measured **latency/power** on real Orin engines but
explicitly did **not** measure accuracy on-device (`summary.json`:
*"DLA INT8 = implicit --int8 … accuracy not claimed"*, and the iGPU numbers were
latency-only). This closes that gap: it runs the **saved `.plan` engines** over the
**same 1000-image bundle** the Jetson A78AE CPU-proxy used (commit `49e30ff`), so every
per-image prediction is 1:1 comparable.

The scientific hook — stage4 found INT8 predictions are **integer-kernel-path dependent**
*among CPUs* (Jetson↔Pi5 bit-identical on the same MLAS SDOT kernel; Jetson↔x86 958/1000
on a different kernel; FP32 always 100%). Here we cross a much bigger datapath boundary:
**CPU integer kernel (MLAS SDOT) vs GPU/DLA integer kernel (TensorRT)**, on **one board**,
with **identical QDQ scales** — `rn50_gpu_int8.plan` was built from
`resnet50_int8_qdq.onnx`, the *same* QDQ ONNX the CPU proxy fed to ORT. So the only
variable under test is the integer datapath.

- Board: **NVIDIA Jetson AGX Orin Developer Kit (64GB)** · JetPack 6.2.1 (L4T R36.4.3) · CUDA 12.6 · **TensorRT 10.3.0** · 2× NVDLA v2 · `nvpmodel MAXN`
- Runner: **`tensorrt` Python API + `cuda-python` 12.9.7** (`from cuda.bindings import runtime as cudart`) — batch1, `execute_async_v3`, deterministic re-run verified
- Preprocess: `÷255 → NHWC→NCHW → ImageNet norm`, **byte-identical to `cpu_proxy/rpi_bench.py`**
- Eval set: `rpi_sub_u8.npy` (1000×224×224×3 u8) + `rpi_labels.npy` — the same bundle on the board and behind the CPU-proxy JSONs

## How to reproduce

```bash
# on the board (~/orin_bench), engines/*.plan already built by the solo sweep
python3 orin_accuracy.py            # runs 5 engines, writes accuracy/rn50_*_accuracy.json (+ pred_cls)
# on the host, after scp'ing accuracy/*.json + rpi_labels.npy into results/
~/emb-ai/bin/python scripts/analyze_accuracy.py    # cross-compares vs cpu_proxy, writes accuracy_summary.json
```

## Results (SSOT = `results/accuracy_summary.json`)

### On-device top-1 (n=1000, same bundle)
| engine | precision | top-1 | note |
|---|---|---:|---|
| `rn50_gpu_fp32` | iGPU FP32 | **0.7620** | = CPU-proxy FP32 |
| `rn50_gpu_fp16` | iGPU FP16 | **0.7620** | 1 flip vs FP32 |
| `rn50_gpu_int8` | **iGPU INT8 (explicit QDQ)** | **0.7620** | **accuracy-valid; = FP32, and > CPU MLAS INT8 (0.750)** |
| `rn50_dla_fp16` | DLA FP16 | **0.7610** | 2 flips vs iGPU FP16 |
| `rn50_dla_int8` | DLA INT8 (implicit `--int8`) | **0.0170** | auto-range, **accuracy NOT claimed** — quantifies the caveat |
| *(cpu-proxy)* | A78AE CPU FP32 (MLAS) | 0.7620 | commit 49e30ff |
| *(cpu-proxy)* | A78AE CPU INT8 (MLAS SDOT) | 0.7500 | commit 49e30ff |

### Prediction agreement (/1000)
| pair | agree | comment |
|---|---:|---|
| iGPU FP32 vs CPU FP32 | **1000** | FP32 bit-identical across the CPU↔GPU boundary (stage4's 100% extends to accelerators) |
| **iGPU INT8 (TRT) vs CPU INT8 (MLAS SDOT)** | **961** | **headline: 39 flips on identical QDQ scales — path dependence crosses CPU↔accelerator** |
| iGPU INT8 vs iGPU FP32 | 943 | 57 flips, yet top-1 identical (net-neutral) — **and 943≠1000 proves the INT8 engine is not a silent FP32 fallback** |
| iGPU FP16 vs iGPU FP32 | 999 | FP16 ≈ lossless |
| DLA FP16 vs iGPU FP16 | 998 | DLA fp16 datapath differs by 2 |
| DLA FP16 vs iGPU FP32 | 999 | |

## Findings

### Finding 1 (headline) — INT8 path dependence crosses the CPU↔accelerator boundary
TRT INT8 (GPU integer kernel) and MLAS INT8 (CPU SDOT kernel) run **identical QDQ scales**
on the **same Orin silicon**, yet disagree on **39/1000 images (96.1% agree)**. That 96.1%
sits in the *same band* as the stage4 Jetson↔x86 CPU cross (958/1000 = 95.8%), **not** the
Jetson↔Pi5 100% (same MLAS SDOT). So the rule generalizes cleanly: **different integer
kernel ⇒ ~96% agreement, regardless of whether the boundary is CPU↔CPU or CPU↔accelerator**;
same kernel ⇒ 100%. INT8 top-1 is *not* reproducible bit-for-bit across datapaths — only
FP32 is (1000/1000 here).

### Finding 2 — the accuracy-valid INT8 path holds on-device (and beats the CPU kernel)
On-device iGPU INT8 (explicit QDQ, real stage3 scales) top-1 = **0.7620 = FP32**, i.e.
lossless at 1000-image granularity — and **higher than CPU MLAS INT8 (0.7500)**. On the 39
headline disagreements, **TRT is right 15×, MLAS right 3×, both wrong 21×** (net +12 =
0.762−0.750). The two kernels apply the same scales but round/accumulate differently, and
here TensorRT's integer datapath is the more faithful one. *(Not a silent FP32 fallback: the
INT8 engine disagrees with its own FP32 on 57 images, and stage3 already showed a distinct
INT8 latency 1.01 ms vs FP32 1.94 ms.)*

### Finding 3 — "top-1 unchanged" ≠ "predictions unchanged"
iGPU INT8 matches iGPU FP32 top-1 exactly (0.762) but flips **57/1000 predictions**; the
flips are net-neutral against the labels. Reporting only top-1 hides real per-image churn —
the pred_cls arrays are the honest artifact.

### Finding 4 — FP16 is effectively lossless; DLA fp16 datapath differs by 2
iGPU FP16 = 999/1000 vs FP32 (one flip). DLA FP16 = 0.7610 (998/1000 vs iGPU FP16) — the
NVDLA v2 fp16 datapath is a hair different from the iGPU's, as expected for a distinct
accelerator, but accuracy-equivalent.

### Finding 5 — implicit DLA INT8 collapses to 0.017 (the caveat, quantified)
`rn50_dla_int8.plan` was built with `trtexec --int8` and **no calibration data** → auto
dynamic-range → top-1 **0.0170** (near-random). The *same harness* gives DLA FP16 = 0.761,
so this is not a harness bug — it is exactly why the solo sweep labeled DLA/implicit INT8
**latency-valid, accuracy-not-claimed**. The accuracy-valid INT8 engine is the explicit-QDQ
iGPU one (Finding 2).

## Caveats
- Absolute top-1 is on a **1000-image subset** (stage1 함정 0: subset top-1 is inflated vs
  the full 50k), so 0.762 is *not* comparable to stage3's 5000-image RTX number — only the
  **same-bundle relative** relations (agreements, INT8-vs-FP32, TRT-vs-MLAS) are the result.
- `dla_int8` (0.017) is **implicit auto-range** — reported only to quantify the
  accuracy-not-claimed caveat, never as a DLA INT8 accuracy.
- Predictions are argmax top-1 only; no calibration re-run — QDQ scales are frozen from the
  stage3 `resnet50_int8_qdq.onnx`.
- Jetson is NVIDIA edge, not one of the three automotive vendors (TI/Qualcomm/Renesas).

## Files
```
scripts/orin_accuracy.py      # on-board TRT runner (tensorrt + cuda.bindings.runtime), dumps pred_cls
scripts/analyze_accuracy.py   # host-side cross-compare vs cpu_proxy, writes accuracy_summary.json
results/accuracy_summary.json # SSOT (top-1 + agreements + headline disagreement breakdown)
results/rn50_gpu_fp32_accuracy.json   # per-engine top-1 + pred_cls (1000)
results/rn50_gpu_fp16_accuracy.json
results/rn50_gpu_int8_accuracy.json
results/rn50_dla_fp16_accuracy.json
results/rn50_dla_int8_accuracy.json
results/orin_accuracy_meta.json       # board/trt/cuda-python provenance
results/rpi_labels.npy                # the 1000 labels (pulled from board for host-side recompute)
raw/orin_accuracy.out                 # on-board stdout (deterministic re-run)
```
