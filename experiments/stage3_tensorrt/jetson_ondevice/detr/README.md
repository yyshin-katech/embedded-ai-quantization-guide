# DETR-on-Orin — new-model axis over the ResNet50 latency ladder

Real on-device TensorRT run of **`facebook/detr-resnet-50`** (transformer detector)
on the **NVIDIA Jetson AGX Orin Dev Kit (64GB)** — same board, same `trtexec`, same
timing flags as the ResNet50 ladder in `../results/summary.json`, so latency is 1:1
comparable. This is the "new model axis": what changes when the model is a
transformer instead of a clean CNN.

- Board: JetPack 6.2.1 (L4T R36.4.3) · CUDA 12.6 · **TensorRT 10.3.0.30** · cuDNN 9.3.0 · Ampere iGPU · 2× NVDLA v2 · `nvpmodel MAXN`
- `trtexec` = `/usr/src/tensorrt/bin/trtexec` (banner `v100300`)
- Timing = `GPU Compute Time` median (device event-timed, H2D/D2H excluded), **batch1**
- Flags (identical to ResNet50): `--warmUp=2000 --duration=10 --iterations=200 --avgRuns=100`
- ONNX: `detr_sim.onnx` (FP32, opset17, fixed `pixel_values [1,3,800,1066]`, 635 nodes) · `detr_int8.onnx` (ORT `quantize_static` QInt8 QDQ, per-channel)

## How to reproduce

```bash
# on the board (~/orin_bench), after scp'ing detr_sim.onnx + detr_int8.onnx to onnx/
python3 detr_bench.py          # builds + times 6 configs, writes detr_results/detr_summary.json
```

`scripts/detr_bench.py` builds+times each config, tees every `trtexec` invocation to
`detr_raw/<tag>.log`, and for the DLA configs additionally runs
`--skipInference --verbose` to capture layer placement.

## Results (SSOT = `results/detr_summary.json`)

| config | onnx | build | latency (GPU-compute median) | vs FP32 | throughput | engine |
|---|---|---|---:|---:|---:|---:|
| iGPU FP32 | detr_sim | ✅ | **25.8496 ms** | ×1.00 | 38.58 qps | 160.12 MiB |
| iGPU FP16 | detr_sim | ✅ | **13.2754 ms** | ×1.947 | 75.21 qps | 81.43 MiB |
| iGPU INT8 (explicit QDQ) | detr_int8 | ❌ **parse fail** | — (stage3 case C) | — | — | — |
| iGPU INT8 (implicit `--int8`) | detr_sim | ✅ | **9.42871 ms** | ×2.742 | 105.91 qps | 58.76 MiB |
| DLA FP16 (`--allowGPUFallback`) | detr_sim | ✅ | **398.64 ms** | ×0.065 | 2.50 qps | 150.45 MiB |
| DLA INT8 (implicit) | detr_sim | ✅ | **78.4756 ms** | ×0.329 | 12.68 qps | 66.34 MiB |

### Finding 1 — heavy transformer ⇒ INT8 arithmetic gain returns (model-dependent)
DETR FP16 = ×1.947, INT8(implicit) = **×2.742** (FP16→INT8 a further ×1.408). On the
same small Ampere iGPU, ResNet50@224 was launch/memory-bound so INT8≈FP16 (ratio
0.984, *no* gain). DETR@800×1066 with big attention matmuls is compute-bound, so INT8
pulls clearly ahead. **"Small Ampere iGPU sees no INT8 gain" is model-dependent** — it
was a property of the light CNN, not the iGPU.

### Finding 2 — explicit-QDQ INT8 does not parse (stage3 case C, on a real transformer)
`detr_int8.onnx` (ORT QInt8: zp≠0 in 1085/1485 Q/DQ + INT32 bias DQ on all 149
Conv/Gemm) fails `trtexec` direct parse in **0.051 s** at node 0
(`/model/Tile_output_0_DequantizeLinear`):

```
Assertion failed: shiftIsAllZeros(zeroPoint): TensorRT only supports symmetric
quantization. The zero point for the QuantizeLinear/DequantizeLinear operator must
be all zeros.
```

This is exactly stage3's **case C** (zp≠0 → `shiftIsAllZeros`), now reproduced on a
real transformer's ORT export. Consequence: on Jetson `trtexec` the **accuracy-valid**
INT8 path is blocked without QDQ surgery / symmetric re-export (the ORT-EP or
`nodes_to_exclude` route from stage3). The **implicit** `--int8` engine builds and
times (9.43 ms) but uses auto dynamic-range ⇒ **latency-only, accuracy NOT claimed**
(same caveat as DLA INT8).

### Finding 3 — a transformer fragments the DLA into uselessness (the headline)
Same `--useDLACore=0 --allowGPUFallback` recipe that gave ResNet50 a clean **2/2**
offload gives DETR:

| placement | ResNet50 | DETR |
|---|---:|---:|
| layers on DLA | 120 | 326 |
| **layers on GPU (fallback)** | **2** | **404** |
| **DLA subgraphs (ForeignNodes)** | **2** | **16** |
| DLA FP16 latency | 17.73 ms | **398.64 ms** (30× slower than iGPU FP16) |

The 404 GPU-fallback layers are the transformer's non-DLA ops: **250 SHUFFLE**
(attention-head reshapes/transposes), 66 CONSTANT, **34 MATRIX_MULTIPLY** (Q·Kᵀ and
score·V — DLA v2 has no dynamic matmul), **30 NORMALIZATION** (LayerNorm — unsupported
on NVDLA v2), 12 UNARY, 12 SELECT (mask `Where`). They split the DLA-eligible work into
**16 islands** (1 = ResNet backbone conv1…input_projection, 4 = encoder self-attn `Mul`
blocks, 11 = FFN `Gemm` blocks), each island boundary a DLA↔GPU tensor copy + sync. 16
round-trips ⇒ 398.64 ms, i.e. **30× slower than simply running FP16 on the iGPU**.
DLA INT8 (78.48 ms) is 5.08× faster than DLA FP16 (the "DLA is an INT8-only machine"
rule still holds on the DLA-resident islands) but still **8.3× slower than iGPU INT8**.

**Design rule (generalized from the concurrent-load finding):** DLA pays off only when
the DLA subgraph has ~0 GPU-fallback layers. A clean CNN (ResNet50: 2 fallback) fits; a
transformer (DETR: 404 fallback, 16 fragments) does not. **DLA is a CNN accelerator; put
transformers on the iGPU.**

## Accuracy (cited, NOT re-measured here)
COCO val2017 is not on the board and the INT8 numbers above are latency-only, so accuracy
is cited from the stage2 RTX 3080 run (`logs/stage2_detr_quantization_report.html`):
FP32 mAP **0.4207** → INT8 (ORT QDQ) **0.2402** (**−42.9%**), small-object mAP_s **−77%**.

## Caveats
- Absolute latency/qps/engine size are event-timed, batch1, MAXN → **relative** relations
  only; not 1:1 comparable to other stages' `polygraphy`/wall-clock numbers.
- iGPU/DLA INT8 here are **implicit** `--int8` (auto range) ⇒ latency/size valid,
  **accuracy not claimed**; the accuracy-valid explicit-QDQ path does not parse (Finding 2).
- DLA fragmentation counts (404/16) are model-specific to DETR's op mix; the *rule*
  (fallback-count gates DLA benefit) generalizes.
- Jetson is NVIDIA edge, not one of the three automotive vendors (TI/Qualcomm/Renesas).

## Files
```
scripts/detr_bench.py                 # build+time+placement harness (runs on board)
results/detr_summary.json             # SSOT
raw/detr_gpu_fp32.log                  # trtexec build+perf logs (6)
raw/detr_gpu_fp16.log
raw/detr_gpu_int8_explicit.log         # the case-C parse failure
raw/detr_gpu_int8_implicit.log
raw/detr_dla_fp16.log
raw/detr_dla_int8_implicit.log
raw/detr_dla_fp16_verbose.log.gz       # --skipInference --verbose placement dumps (gzip: 5.9MB→0.3MB; `gunzip -c` to read)
raw/detr_dla_int8_implicit_verbose.log.gz
raw/detr_bench.out                     # harness stdout
```
The two `*_verbose.log.gz` are the full `--skipInference --verbose` layer dumps
(gzipped losslessly — 5.9 MB / 10.8 MB raw → 0.3 MB / 0.5 MB); the placement counts
in Finding 3 (326 DLA / 404 GPU / 16 ForeignNodes) are re-countable from them via
`gunzip -c raw/detr_dla_fp16_verbose.log.gz | grep -c '\[GpuLayer\]'` etc.
