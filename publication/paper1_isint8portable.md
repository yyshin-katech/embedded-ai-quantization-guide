# Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on Embedded and Automotive Accelerators

> **DRAFT — v0.3 (arXiv target).** Empirical sections wired from the repository's SSOT (30 measurement reports in `logs/`, 9 experiment suites in `experiments/`). Related Work (§2) and References are complete: all 46 BibTeX entries were verified against their arXiv/publisher pages (titles, authors, venues) on 2026-09-04. All numbers are relative comparisons under the caveats in §11. No internal infrastructure identifiers are included. Authorship is a placeholder for the submitting author(s).

**Author(s):** _[Author]_, Korea Automotive Technology Institute (KATECH) · _[co-authors TBD]_
**Contact:** _[email]_
**Status:** pre-submission draft · target: arXiv (cs.LG / cs.PF / cs.AR)

---

## Abstract

Eight-bit integer (INT8) post-training quantization is the default recipe for deploying deep networks to the edge, under a widely held assumption: enabling INT8 makes inference faster at a small, predictable accuracy cost, and a model quantized once can be carried to any target. We test that assumption directly with a controlled cross-platform measurement study spanning seven hardware classes — ARM and x86 CPUs, an RTX-class discrete GPU, an NVIDIA Jetson AGX Orin iGPU and its NVDLA cores, and two vendor NPUs (Qualcomm Hexagon HTP and a DEEPX DX-M1 M.2 accelerator) — using a single ONNX artifact and identical quantization scales wherever possible so that the integer kernel or ISA is the only free variable. Portability fails on three independent axes. **(1)** The *sign* of the INT8 speedup is determined by whether the CPU has dot-product instructions (ARM `dotprod`/SDOT, x86 VNNI): dot-product cores speed up by up to 2.1×, while cores lacking them slow down by 1.6–1.8×, for the identical model and runtime. **(2)** INT8 numerical outputs are not portable: FP32 predictions are bit-identical across every platform pair (1000/1000), but INT8 predictions *disagree* across CPU↔CPU (958/1000), across the CPU↔accelerator boundary (961/1000 for a GPU integer kernel vs. a CPU integer kernel built from the same scales), and across CPU↔vendor-NPU (939/1000), even though top-1 accuracy is preserved — the disagreement is integer-kernel path-dependence, not accuracy loss. **(3)** Vendor NPUs own quantization: a bring-your-own QDQ graph is rejected in two opposite ways — silently (Qualcomm HTP ignores external scales and collapses accuracy from 0.75 to 0.005 while appearing to run) or loudly (the DEEPX compiler refuses the graph outright) — and only the vendor's native quantization path produces a correct, fast engine. We further show that edge-NPU latency *regimes* are set by model output / device-to-host transfer size, not by compute: the same accelerator is compute-bound, data-movement-bound, or host-bound depending only on the model's output tensor. We release the measurement scripts and 30 reports. The results argue that "quantize once, deploy anywhere" is unsafe for embedded and automotive deployment, where per-input determinism and redundancy matter.

---

## 1. Introduction

Quantizing a floating-point network to 8-bit integers is the first and often only compression step in an embedded deployment pipeline. The operational folklore is compact and appealing: *(i)* INT8 is faster than FP32/FP16 because integer math is cheaper and moves less memory; *(ii)* the accuracy cost is small and bounded; and *(iii)* a model quantized once is a portable artifact — an ONNX file with QDQ nodes, or a TFLite model — that can be handed to any runtime or accelerator. This folklore underlies the way quantization is taught, benchmarked (report one speedup and one accuracy number), and shipped.

This paper asks whether the folklore survives contact with real, heterogeneous hardware. We ran a controlled measurement campaign across seven hardware classes and three model families (image classification, CNN object detection, and transformer detection), taking care to hold the model artifact and — critically — the quantization *scales* fixed across targets so that any difference we observe is attributable to the target's integer kernel or instruction set, not to a different quantizer.

We find that all three parts of the folklore fail, and they fail in ways that matter specifically for embedded and automotive systems:

1. **The speedup can be negative, and its sign is an ISA property (§4).** For the identical INT8 model on the identical CPU runtime (ONNX Runtime, MLAS), a Raspberry Pi 5 Cortex-A76 and a Jetson A78AE — which have ARM dot-product instructions — get 1.83× and 2.11× *faster*, while an x86 Core i9 without AVX-512 VNNI and a Cortex-A53 without dot-product get 1.65–1.76× *slower*. INT8 is not "faster"; it is faster *iff* the target has the right accumulate instruction.

2. **INT8 outputs are not numerically portable (§5, headline).** FP32 predictions are bit-identical across all platform pairs we tested (1000/1000). INT8 predictions are not: they disagree on ~4% of inputs across a CPU↔CPU pair, across the CPU↔GPU boundary (built from the *same* QDQ scales), and across a CPU↔vendor-NPU pair, with the disagreement growing as the kernels diverge (1000 → 961 → 939 agreements out of 1000). Top-1 accuracy is preserved — the flips are net-neutral — so this is invisible to a standard accuracy report, yet it means a quantized model is not a deterministic function of its input once you change the target. Concurrent work localizes the underlying epilogue-rounding mechanism by swapping INT8 kernels on a *single* GPU for LLMs [chen2026integeralibi]; our contribution is orthogonal and, for deployment, more consequential — the cross-*physical-device* measurement for vision/detection models under an identical ONNX file and identical scales, with FP32 bit-identity as a control (§5).

3. **Vendors own quantization (§6).** A model you quantized yourself is not deployable to a vendor NPU as-is. Qualcomm's Hexagon HTP *silently* ignores external QDQ scales and collapses accuracy (0.75 → 0.005) while compiling, profiling, and running without error; the DEEPX compiler *loudly* rejects the same class of graph. Both are correct only through the vendor's own quantization path. These are opposite symptoms of one fact: the accelerator, not your toolchain, defines the numerics.

Beyond portability, we contribute a systems observation that reframes edge-NPU performance analysis: **latency regime is set by output/data-movement size, not compute (§7).** On one M.2 NPU, a classifier with a 4 KB output is compute-bound and scales 2.19× across cores, while a detector with a 2.82 MB raw output on the same device, same runtime, is data-movement-bound and does not scale at all — and a *lighter*-compute detector with an even larger output is 26× slower still. We isolate output size as the causal variable.

We frame these findings for their intended audience. The submitting institution works on automotive edge AI, where the failures above are not academic: non-portable INT8 numerics undermine the determinism and cross-module consistency that safety cases and redundant (dual-compute) architectures rely on. §8–§10 add supporting characterization (transformer INT8 breakdown, DLA behavior) and a catalog of silent-failure pitfalls we hit, and §11 states the study's limits honestly.

**Contributions.**
- A controlled, same-artifact/same-scale cross-platform methodology that isolates the integer kernel/ISA as the only variable (§3).
- C1: the INT8 speedup sign is determined by the CPU dot-product ISA, shown across four CPUs (§4).
- C2 (headline): INT8 numerical non-portability across CPU↔CPU and CPU↔accelerator boundaries, with FP32 as a bit-identical control (§5).
- C3: vendor NPUs own quantization; two opposite BYO-QDQ failure modes (§6).
- C4: edge-NPU bottleneck regimes are set by output/data-movement size, not compute (§7).
- A reproducibility artifact: scripts plus 30 measurement reports.

---

## 2. Background and Related Work

**Quantization foundations.** Integer-arithmetic-only inference — mapping weights and activations to INT8, accumulating products in INT32, and re-scaling back — was formalized by Jacob et al. [jacob2018] and codified for post-training use in the whitepapers of Krishnamoorthi [krishnamoorthi2018] and Nagel et al. [nagel2021whitepaper], with a broad survey by Gholami et al. [gholami2021survey]. The per-tensor vs. per-channel and symmetric vs. asymmetric design axes, and the result that symmetric INT8 typically stays within ~1% of FP32 for CNNs, are established by NVIDIA's evaluation [wu2020]; rounding choice at quantization time is itself accuracy-critical [nagel2020adaround]. Our work proposes no new quantization algorithm — it measures how *existing*, correctly-produced INT8 behaves once it crosses a hardware boundary. Crucially, Jacob's pipeline makes the INT32 accumulate exact and order-independent; the re-scaling *epilogue* is where implementation freedom — and, as we and concurrent work show, non-portability — lives.

**Integer kernels and dot-product ISAs.** The INT8 speed advantage depends on hardware dot-product/accumulate instructions: ARM `dotprod` (SDOT/UDOT, ARMv8.2-A) [armisa] and x86 AVX-512 VNNI [intelisa], exploited by the low-precision GEMM libraries gemmlowp [gemmlowp], XNNPACK [xnnpack], FBGEMM [khudia2021fbgemm], and Microsoft's MLAS [mlas] (the kernel library our ONNX Runtime CPU path uses). Prior characterization of data-center INT8 inference [park2018facebook] documents this dependence for throughput. We contribute the *cross-device sign-flip* framing — that the same model and runtime is faster or slower depending only on whether the target CPU has these instructions — and connect it to the accuracy-side consequence (§4 → §5).

**Edge and mobile inference benchmarking.** MLPerf Inference [reddi2020mlperf], MLPerf Tiny [banbury2021mlperftiny], MLPerf Mobile [reddi2020mobile], and AI-Benchmark [ignatov2018, ignatov2019] are the standard cross-device benchmarks, and recent studies benchmark detectors across the exact hardware family we use (Jetson, Raspberry Pi 5, Coral) [edgedetection2024, millar2025]. By design these quantize *per submission and per backend* and report per-device throughput/accuracy scores; none holds the quantization scales fixed across targets, and none reports cross-target numerical *agreement*. That axis — numerical portability under fixed scales — is our C2.

**Numerical reproducibility and determinism (closest to our headline).** The mechanism behind C2 was, concurrently with this work, localized by Chen [chen2026integeralibi, chen2026deterministic]: swapping only the INT8 GEMM kernel (CUTLASS vs. Triton) inside an LLM serving stack *on a single GPU* yields two engines that are each bit-reproducible against themselves yet agree on no generated sequence, with the divergence traced to scale application and output rounding in the *epilogue* (the INT32 accumulate being exact), and power-of-two scales restoring bit-identical cross-kernel agreement. We cite this as concurrent prior work and claim no discovery of the mechanism. Our contribution is orthogonal in setting and method: Chen swaps kernels on one device for LLMs, whereas we measure per-input prediction disagreement *across physical hardware boundaries* — dot-product CPU ↔ non-dot-product CPU, CPU ↔ GPU, and CPU ↔ vendor NPU — for vision and detection models, using an identical ONNX file with identical embedded QDQ scales, and we contrast the INT8 divergence against FP32 bit-identity on those same devices. More broadly, MQBench [li2021mqbench] measures a hardware-deployability *accuracy gap* across backends but not per-input, bit-level disagreement under identical scales; the inference backend has been shown to confound even greedy-decoding LLM behavior [masoudian2026]; and floating-point non-associativity [fpnonassoc2024] and fixed-reduction-order remedies [repdl2025] frame the FP side — we scope our "FP32 bit-identical" claim to our observed, fixed-thread configuration accordingly (§5, §11). The divergence is folklore in practitioner issue trackers for quantized TFLite CPU-vs-NPU and cross-EP ONNX Runtime outputs, but to our knowledge has not been systematically measured across embedded and automotive accelerators.

**Transformer quantization.** Transformer INT8 fragility is driven by activation outliers [dettmers2022llmint8, bondarenko2021], addressed by activation migration (SmoothQuant [xiao2022smoothquant]), activation-aware weight scaling (AWQ [lin2023awq]), and weight-only PTQ (GPTQ [frantar2022gptq]); for vision transformers specifically, PTQ4ViT [yuan2022ptq4vit] and Liu et al. [liu2021ptqvit] handle post-softmax/GELU activations. We use these to explain *why* DETR INT8 collapses on-device and to quantify how much the activation-granularity lever recovers on a real toolchain vs. in fake-quant (§8) — reinforcing that activations, not op selection, are the fragile axis.

**Vendor NPU toolchains.** Vendor compilers differ in scaling, clipping, and kernel support, so the same checkpoint yields inconsistent cross-backend accuracy — a point made by MQBench [li2021mqbench] and, most directly, by Quant-Trim [dhahri2025quanttrim], which proposes a training-time hardware-neutral checkpoint. Qualcomm's QNN/AI-Hub stack quantizes to its own native format [qualcomm_qnn], as do Apple Core ML [coreml] and TFLite/LiteRT delegates [litert]. These works establish that vendors prefer their own quantization; we contribute a controlled, cross-vendor account of *bring-your-own-QDQ rejection* and its two opposite failure modes — silent (accuracy collapse while running) vs. loud (compile refusal) — and show the native path is both correct and faster (§6).

**Accelerator characterization.** The compute-bound vs. memory-bound dichotomy is the Roofline model [williams2009roofline]; the primacy of data movement over compute energy is the Eyeriss line of work [chen2016eyeriss, sze2017efficient]. We extend the deployment-level picture with a third, output-size-driven regime — device-to-host (D2H)/PCIe-bound — and show that on one PCIe-attached edge NPU the bottleneck (and whether multi-core helps) is set by the model's output tensor size, not its compute (§7). Concurrent-inference profiling on Jetson [jetsonconcurrent2025] corroborates our related finding that GPU-fallback subgraphs serialize otherwise-parallel accelerator work.

**NVDLA and fixed-function INT8 accelerators.** The Orin DLA is an NVDLA v2 instance [nvdla, farshchi2019nvdla]; we characterize it as an INT8-only, CNN-favoring datapath that is the perf-per-watt leader for CNNs but fragments on transformers (§8).

**Automotive compute and redundancy (framing).** Redundant, diverse compute across CPU/GPU/DLA is the backbone of automotive functional-safety architectures (ISO 26262 / ASIL [iso26262], NVIDIA DRIVE [nvidiadrive]) and heterogeneous AV-SoC scheduling [hetsched2022]. This is our motivation: non-portable INT8 numerics (§5) directly threaten the cross-module agreement that dual-compute redundancy assumes — the concern that motivates our planned follow-on work on a multi-module platform (§12).

---

## 3. Experimental Methodology

### 3.1 Hardware matrix

| Class | Target | Role in the study |
|---|---|---|
| ARM CPU (dotprod) | Raspberry Pi 5, Cortex-A76 | C1 sign, C2 agreement |
| ARM CPU (dotprod) | Jetson AGX Orin, Cortex-A78AE | C1 sign, C2 CPU↔accelerator |
| ARM CPU (no dotprod) | i.MX8M-Nano, Cortex-A53 | C1 sign (negative) |
| x86 CPU (no VNNI) | Core i9-10900K | C1 sign (negative), C2 |
| Discrete GPU | RTX 3080 (Ampere) | precision ladder, transformer INT8 |
| Edge iGPU + NVDLA | Jetson AGX Orin (iGPU, 2×NVDLA v2) | accelerator char., C2 GPU kernel |
| Vendor NPU (mobile/auto) | Qualcomm Hexagon HTP (QCS8550, SA8775P) | C3 (silent BYO-QDQ) |
| Vendor NPU (auto) | DEEPX DX-M1 (M.2, PCIe Gen2×1) | C3 (loud BYO-QDQ), C4 regimes |

Every device is a single unit (n=1 per class); we therefore make *relative* claims about representative devices, not population claims about an ISA (see §11).

### 3.2 Models and datasets

ResNet-18/50 (ImageNet-1k classification), DETR-ResNet-50 (COCO detection, transformer), YOLO26n and YOLOv5s (COCO detection, CNN), plus BEVFormer/BEVDet (3D BEV) used only for latency/engine characterization. Accuracy is reported on ImageNet val (subset or full, stated per result) and COCO val2017 subsets. Absolute accuracy/latency are not comparable across sections because batch size, input resolution, and evaluation subset differ; we report *relative* deltas within a controlled comparison.

### 3.3 Controlled-comparison principle

The core methodological device of this paper: for any cross-target comparison, we fix the model artifact and the quantization *scales*, so the only free variable is the target's integer kernel/ISA. Concretely, the same `resnet50_int8_qdq.onnx` (with its embedded QDQ scales) is (a) run on multiple CPU EPs and (b) used to *build* the TensorRT INT8 engine — so the GPU integer kernel and the CPU integer kernel consume identical scales. When a comparison cannot hold scales fixed (e.g., a vendor NPU that rejects external QDQ), we say so and treat the result as a deployment finding, not a kernel comparison.

### 3.4 Measurement protocol and scope

Latencies are event-timed on GPU/accelerator paths and wall-clock on CPU/harness paths (the two are not directly comparable and are never mixed within a claim). Unless noted, batch size is 1. Numerical agreement is reported as the number of inputs (out of a fixed bundle, typically 1000 for classification) on which two targets produce the *same* top-1 prediction. **Known limitation:** most latencies are single-run p50 without confidence intervals, and several accuracy numbers are on subsets; §11 quantifies why this bounds our claims to relative comparisons, and this study's own §9 shows subset evaluation can inflate top-1 by ~9.77 percentage points.

---

## 4. The Speedup Sign Is ISA-Determined (C1)

We ran the identical ResNet-50 INT8 QDQ model on the identical runtime (ONNX Runtime, CPU execution provider, MLAS kernels) on four CPUs, and compared against the same model in FP32 on each.

| CPU | Dot-product ISA | FP32 → INT8 | INT8 effect |
|---|---|---|---|
| Cortex-A76 (Raspberry Pi 5) | ARM `dotprod` (SDOT) | 144.95 → 79.08 ms | **1.83× faster** |
| Cortex-A78AE (Jetson AGX Orin) | ARM `dotprod` | 38.47 → 18.22 ms | **2.11× faster** |
| Core i9-10900K (x86) | no AVX-512 VNNI | 9.28 → 16.34 ms | **1.76× slower** |
| Cortex-A53 (i.MX8M-Nano) | no dot-product | — | **1.65× slower** |

The determinant is not the ISA *family* (both ARM and x86 appear on both sides): the A53 is ARM yet slows down, the A76/A78AE are ARM and speed up. The determinant is the presence of a dot-product/accumulate instruction (ARM `dotprod`, x86 VNNI). Without it, INT8's re-quantization and widening overhead is not amortized by a faster inner product, and INT8 is a *pessimization*. This has an immediate practical consequence: a fleet of heterogeneous edge CPUs cannot assume INT8 is a win; the same binary regresses on the wrong core.

---

## 5. INT8 Outputs Are Not Portable (C2) — headline

If two targets run the *same* quantized model with the *same* scales, do they produce the same predictions? For FP32 the answer is yes, exactly. For INT8 it is no.

**FP32 is a bit-identical control.** Across every platform pair we tested, FP32 top-1 predictions agree on 1000/1000 inputs. Whatever divergence we see under INT8 is therefore not a floating-point reduction-order artifact of our harness; it is specific to the integer path. We scope this bit-identity to our observed configuration (fixed thread count, a single reduction path per target): floating-point non-associativity can make FP reductions non-deterministic under different parallelization or hardware [fpnonassoc2024], so the precise claim is "FP32 was bit-identical across these targets as measured," not that FP32 is portable by construction. The contrast we rely on — FP32 identical, INT8 diverging, on the *same* devices and harness — holds regardless.

**INT8 disagreements, ordered by kernel divergence.**

| Comparison | Boundary | INT8 agreement | Note |
|---|---|---|---|
| Jetson A78AE vs. Pi5 A76 | CPU↔CPU, same MLAS SDOT kernel | **1000 / 1000** | identical integer path → identical outputs |
| Raspberry Pi 5 vs. x86 i9 | CPU↔CPU, different integer path | **958 / 1000** | different kernels → ~4% disagree |
| Jetson iGPU (TensorRT INT8) vs. A78AE (MLAS INT8) | **CPU↔accelerator**, same QDQ scales | **961 / 1000** | crosses the accelerator boundary |
| DEEPX DX-M1 (NPU) vs. host A76 CPU | **CPU↔vendor-NPU** | **939 / 1000** | crosses to a third-party NPU |

Two things make this a clean result. First, the CPU↔GPU comparison uses the *same* `resnet50_int8_qdq.onnx` to build the TensorRT engine that the CPU EP runs, so the QDQ scales are identical and the only variable is the integer kernel (TensorRT's INT8 kernels vs. MLAS's) — yet they disagree on 39/1000 inputs. Second, when two targets *do* share an integer path (A78AE and A76 both using MLAS SDOT), agreement returns to a perfect 1000/1000, confirming the divergence is kernel-specific rounding/accumulation, not hardware noise.

**Accuracy hides it.** The disagreements are net-neutral: on the Jetson iGPU, accuracy-valid INT8 top-1 is 0.7620 — *lossless* versus its own FP32 and higher than the CPU MLAS INT8 top-1 of 0.7500 — even though the two INT8 paths flip predictions on dozens of individual inputs. A standard "accuracy after quantization" report would show no problem. The portability failure is only visible when you compare *predictions per input across targets*.

**Relation to concurrent work.** The mechanism behind these disagreements — divergence introduced in the re-quantization *epilogue* (scale application and output rounding) after an exact INT32 accumulate — was localized concurrently by Chen [chen2026integeralibi, chen2026deterministic], who swapped INT8 GEMM kernels (CUTLASS vs. Triton) *on a single GPU* for LLMs and showed that power-of-two scales restore bit-identical cross-kernel agreement. We do not claim to discover this mechanism, and we cite that work as concurrent prior art. Our result is complementary and, for deployment, more consequential in three ways: the divergence persists across *different physical devices* (a dot-product CPU vs. a non-dot-product CPU, a CPU vs. a GPU built from the same scales, a CPU vs. a third-party vendor NPU); it holds for vision and detection models rather than LLMs; and it is measured against an FP32 bit-identical control on the same devices. Where Chen asks whether two kernels on one GPU agree, we ask whether a fielded fleet of heterogeneous targets agrees — the question a cross-module automotive platform actually poses.

**Why it matters.** A quantized model is often treated as a deterministic function `f(x)`. C2 says that once you change the target, `f` changes on a measurable fraction of inputs, silently, with no accuracy signal. For automotive systems this is the crux: dual-compute redundancy and cross-module consistency checks assume two units computing the same input agree; C2 shows that assumption holds under FP32 but not under INT8 across heterogeneous integer kernels. The monotone erosion (1000 → 961 → 939) as the second target moves from same-kernel CPU, to accelerator, to vendor NPU, quantifies how much determinism you spend for each step away from the reference kernel.

---

## 6. Vendors Own Quantization: Two Failure Modes (C3)

C1–C2 assume you can even *run* your quantized model on the target. On vendor NPUs, you frequently cannot — the accelerator insists on quantizing the model itself.

**Qualcomm Hexagon HTP — silent.** Submitting an externally quantized ONNX-QDQ ResNet-50 to Qualcomm AI Hub compiles, profiles, and runs on-device with no error, but the HTP ignores the external QDQ scales and on-device top-1 collapses from 0.75 to **0.005**. The same ONNX runs correctly on x86 CPU (0.753), and the FP32/fp16 path on HTP is faithful (0.745) — so the failure is specific to *externally supplied* INT8 scales being discarded. The correct path is the vendor's own `submit_quantize_job` (HTP-native QDQ), which recovers top-1 to **0.735** and is *faster and leaner* than the external-QDQ engine (748 µs vs. 1052 µs).

**DEEPX DX-M1 — loud.** Feeding the same class of externally quantized graph to the DEEPX compiler produces a hard error — `GraphStructureError: 106 isolated node(s)` → `InternalError`, with no engine emitted. The native path (supply FP32; let the DEEPX compiler run its own PTQ) compiles cleanly and reaches top-1 **0.7660**, lossless-grade.

**One root cause, opposite symptoms.** Qualcomm fails *open* (runs a silently broken model — dangerous, because a broken model can ship) and DEEPX fails *closed* (refuses to build — safe, because nothing broken can ship). Both encode the same rule: the accelerator, not your toolchain, owns the quantization. The deployment consequence is that a quantization you validated on one target is not a portable artifact to a vendor NPU at all — reinforcing C2 from the deployability side, and adding a concrete safety-relevant hazard in the Qualcomm case (a silently wrong INT8 model that passes compile and profile).

---

## 7. Bottleneck Regimes Are Set by Output Size, Not Hardware (C4)

Reasoning about "is this NPU fast enough" usually starts from compute (FLOPs/TOPS). On a PCIe-attached edge NPU we find the latency *regime* — and whether adding cores helps at all — is set by the model's output (device-to-host, D2H) transfer size, on the same device and runtime.

| Model (on DEEPX DX-M1) | Output size | Regime | Multi-core scaling |
|---|---|---|---|
| ResNet-50 | 4 KB | **compute-bound** (compute 2.77 ms ≫ D2H 0.11 ms) | 2.19× near-linear on 3 cores |
| YOLO26n | 2.82 MB (raw head) | **D2H-bound** (D2H 21.81 ms ≫ compute 9.0 ms) | 1.00× flat |
| YOLOv5s | 5.48 MB | **D2H-bound**, worse | lightest compute (2.59 ms) yet **26× slower** |

The YOLOv5s row isolates the causal variable: it has the *smallest* compute of the three yet is by far the slowest, because it has the largest output to move across the PCIe Gen2×1 link. Compute does not predict the regime; output/D2H size does.

We observe a third regime with the transformer detector DETR, where the DEEPX compiler auto-splits the graph and leaves the transformer on the host CPU in FP32: end-to-end 1036.34 ms decomposes as host-CPU transformer FP32 910.6 ms (87.9%) ≫ D2H 57.28 ms ≫ NPU INT8 41.11 ms (4.0%) ≫ H2D 6.97 ms — **host-CPU-compute-bound**. Three models on one accelerator thus exhibit three different bottlenecks (NPU-compute, PCIe-D2H, host-CPU-compute). For context, in its favorable (compute-bound) regime the same NPU delivers large wins over the host CPU — e.g., YOLO26n throughput 91.51 fps vs. 8.01 fps on the A76 (×11.42) and host-side perf/watt ×29.29 — but those wins evaporate the moment the model's output pushes it into the D2H-bound regime. The design rule: for edge accelerators behind a bus, provision and partition by data movement, not by TOPS.

---

## 8. When INT8 Breaks Down: Transformers and the Granularity Lever (supporting)

C1–C3 use CNNs, where INT8 is (kernel permitting) nearly lossless. Transformers are the stress test and reinforce C2's thesis that *activations*, not ops, are where INT8 portability breaks.

DETR INT8 collapses hard and reproducibly: FP32 mAP 0.4207 → INT8 0.2402 (−42.9%) on a discrete GPU (ORT dynamic, 5000 images), cross-confirmed on Jetson with symmetric re-quantization (0.4237 → 0.2383, −43.8%), with small-object mAP down 77–85%. Ablation shows the cause is activation *variance/granularity*, not op selection: leaving all 36 attention matmuls in FP barely moves the number (+0.36), while quantizing only the backbone is near-lossless and quantizing only the transformer is what collapses. The activation-granularity lever (SmoothQuant) recovers 59.9% of the gap in a torch fake-quant setting but only ~9% on-device — because the only on-device-buildable INT8 path quantizes Gemms only (attention/LayerNorm/Softmax stay FP16), so the lever cannot reach the dominant residual. Consistently, the DEEPX compiler produces *no* transformer INT8 collapse (mAP 0.4377 → 0.4385, −0.2%) precisely because it declines to quantize the transformer and keeps it on the host in FP32 — "no collapse" and "collapse" are two sides of the same fact: transformer activations do not survive INT8, so a toolchain either refuses (no loss, no speedup) or forces it (speedup, large loss). This is the portability thesis again, in the accuracy dimension.

*(Accelerator note.* On the Jetson NVDLA v2, INT8 is not merely preferred but mandatory: DLA is an INT8-only datapath (FP16 is 13.87× slower) and is the perf-per-watt leader (51.29 inf/s/W, ~1.55× the iGPU at roughly half the power) for CNNs, but fragments on transformers (DETR: 404 GPU-fallbacks / 16 ForeignNodes, 30× slower than the iGPU). The accelerator's "preferred precision" is itself a non-portable, model-dependent property.)*

---

## 9. Methodology Pitfalls and Silent Failures (C8)

The measurements above were only trustworthy after we removed a series of silent errors that a normal pipeline would not surface. We report them because they bound what any single-number quantization result means.

- **Subset evaluation inflates accuracy.** A 1000-image ImageNet subset overstated top-1 by **+9.77 percentage points** versus the full 50k val set, and flipped the sign of three quantization deltas and five significance verdicts. Accuracy claims here are either on full val or explicitly flagged as relative-on-subset.
- **Preprocessing dominates the quantization delta.** The choice of resize/crop (squash vs. torchvision) changed top-1 by **−1.07 pp**, roughly **9×** the −0.12 pp cost of the quantization itself. A quantization number is meaningless without a fixed, reported preprocessing.
- **SQNR does not predict accuracy.** Per-layer SQNR had essentially no rank correlation with the top-1 delta (Spearman −0.04); the common practice of ranking layers by SQNR to guide mixed precision is not supported here.
- **Silent fallbacks are everywhere.** External QDQ silently falling back to CPU/FP32; a "TensorProvider present" that quietly runs on CPU because a library was off the loader path; a zero-copy output buffer aliasing bug that collapsed top-1 to 0.0014 with no error; an opset down-convert that "succeeds" (exit 0) while producing an invalid graph; the pip TensorRT wheel shipping without the `trtexec` binary the tutorials assume. Each of these produces a plausible-looking number that is wrong.

---

## 10. (reserved)

`[If space/scope allows: QAT recovery cost as a short section — W4A8 FP32 68.51 → PTQ 44.35 → QAT 67.81 (97.1% recovery), with a fake-quant-removed control isolating +0.80 pp of pure fine-tuning, leaving an irreducible −1.50 pp "QAT is real but not free." Currently out of scope to keep the portability focus; kept here as a note.]`

---

## 11. Threats to Validity

We state the study's limits plainly; several are properties of a measurement-first project and bound our claims to *relative* comparisons.

- **Single-run latencies without confidence intervals.** Most latencies are p50 or single-run. We do not claim differences smaller than a few percent; the headline results (sign flips of 1.6–2.1×, regime differences of >20×) are far outside plausible run-to-run noise, but the smaller ones should be read as directional.
- **Subset accuracy.** Several accuracy and agreement numbers are on subsets (200–1000 images). This study itself quantifies the risk (§9, +9.77 pp inflation); the agreement counts (958/961/939/1000) are on fixed 1000-image bundles and are internally comparable, but not comparable to full-val absolute accuracy.
- **n=1 per hardware class.** One unit per class. We claim "this representative device," never "all A76" or "all x86."
- **Version confounds.** Runtime versions differ across platforms (e.g., ORT and TensorRT versions differ by target). Where a comparison could be confounded by version (e.g., a cross-run 1000/1000 spanning two ORT versions), we flag it; the same-machine comparisons (the core of C1/C2) are not version-confounded.
- **Relative, not absolute.** Batch size, input resolution, and evaluation subset differ across sections; absolute latency/accuracy are not cross-comparable. All claims are within-comparison relative deltas.
- **Power-measurement gap.** Some perf-per-watt figures use a host-side power boundary because on-board/M.2 card power (upstream of the accessible rail) or DLA power (not captured by the GPU utilization counter) could not be isolated; we report the measurement boundary alongside each figure.
- **Init-weight models excluded from accuracy.** The BEV capstone models ran with initialization weights (public weights unavailable), so their mAP is ~0 by construction and is used only for latency/engine-size characterization, never for accuracy claims.
- **Vendor scope.** Vendor-NPU findings cover Qualcomm and DEEPX; other automotive NPUs (TI, Renesas) were not available and are left to future work.

---

## 12. Conclusion

Across seven hardware classes we find that INT8 quantization is not portable on any of the three axes the operational folklore assumes. Its *speedup* can be negative and its sign is set by the CPU's dot-product ISA. Its *numerics* are not portable: identical scales produce disagreeing predictions across CPU↔CPU and CPU↔accelerator boundaries, while FP32 stays bit-identical — an accuracy-invisible loss of determinism. Its *deployability* is gated by the target vendor, which owns quantization and rejects a bring-your-own QDQ graph either silently (dangerously) or loudly. We add that edge-NPU latency regimes are governed by data movement, not compute.

The practical recommendations are concrete: re-validate INT8 per target rather than once; treat vendor-native quantization as mandatory, not optional; provision accelerators by output/data-movement size; and, for safety-relevant or redundant automotive compute, do not assume two heterogeneous units running the same INT8 model agree per input — they do under FP32 and may not under INT8. Where per-input cross-target determinism is required, constraining quantization to power-of-two scales is a promising mitigation — shown to restore bit-identical cross-kernel agreement in the single-GPU LLM setting [chen2026deterministic] — whose effectiveness across *physical* device boundaries (CPU↔accelerator, CPU↔vendor-NPU, and against vendor-native re-quantization) is, to our knowledge, untested and a concrete next step. These findings also motivate our follow-on work characterizing a heterogeneous multi-module automotive compute platform, where the inter-module data-movement bottleneck (a level up from §7) and cross-module INT8 consistency (a level up from §5) become first-order system design constraints.

**Artifact availability.** Measurement scripts and 30 HTML reports are released with this paper. `[link TBD]`

---

## References

References are provided below as BibTeX (drop into `refs.bib` for the LaTeX build). **Verification status:** all entries were verified against their arXiv abstract pages (titles, full author lists, venues) on 2026-09-04; no `[unverified]` entries remain. The two Chen 2026 entries are **concurrent preprints** central to §5 positioning (*The Integer Alibi* is a companion to arXiv:2608.11693); re-check for any updated or formally published version immediately before submission.

```bibtex
% ---- Quantization foundations ----
@inproceedings{jacob2018,
  title     = {Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  author    = {Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2704--2713},
  year      = {2018},
  note      = {arXiv:1712.05877}
}
@misc{krishnamoorthi2018,
  title  = {Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper},
  author = {Krishnamoorthi, Raghuraman},
  year   = {2018},
  note   = {arXiv:1806.08342}
}
@misc{nagel2021whitepaper,
  title  = {A White Paper on Neural Network Quantization},
  author = {Nagel, Markus and Fournarakis, Marios and Amjad, Rana Ali and Bondarenko, Yelysei and van Baalen, Mart and Blankevoort, Tijmen},
  year   = {2021},
  note   = {arXiv:2106.08295}
}
@misc{gholami2021survey,
  title  = {A Survey of Quantization Methods for Efficient Neural Network Inference},
  author = {Gholami, Amir and Kim, Sehoon and Dong, Zhen and Yao, Zhewei and Mahoney, Michael W. and Keutzer, Kurt},
  year   = {2021},
  note   = {arXiv:2103.13630}
}
@misc{wu2020,
  title  = {Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation},
  author = {Wu, Hao and Judd, Patrick and Zhang, Xiaojie and Isaev, Mikhail and Micikevicius, Paulius},
  year   = {2020},
  note   = {arXiv:2004.09602}
}
@inproceedings{nagel2020adaround,
  title     = {Up or Down? Adaptive Rounding for Post-Training Quantization},
  author    = {Nagel, Markus and Amjad, Rana Ali and van Baalen, Mart and Louizos, Christos and Blankevoort, Tijmen},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020},
  note      = {arXiv:2004.10568}
}

% ---- Integer GEMM kernels & dot-product ISAs ----
@misc{khudia2021fbgemm,
  title  = {FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference},
  author = {Khudia, Daya and Huang, Jianyu and Basu, Protonu and Deng, Summer and Liu, Haixin and Park, Jongsoo and Smelyanskiy, Mikhail},
  year   = {2021},
  note   = {arXiv:2101.05615}
}
@misc{park2018facebook,
  title  = {Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications},
  author = {Park, Jongsoo and Naumov, Maxim and Basu, Protonu and others},
  year   = {2018},
  note   = {arXiv:1811.09886}
}
@misc{gemmlowp,
  title        = {{gemmlowp}: A Small Self-Contained Low-Precision {GEMM} Library},
  author       = {{Google}},
  howpublished = {\url{https://github.com/google/gemmlowp}},
  note         = {Accessed 2026}
}
@misc{xnnpack,
  title        = {{XNNPACK}: Optimized Floating-Point and Quantized Neural Network Inference Operators},
  author       = {{Google}},
  howpublished = {\url{https://github.com/google/XNNPACK}},
  note         = {ARM DotProd (SDOT / i8mm) INT8 path; accessed 2026}
}
@misc{mlas,
  title        = {{MLAS}: Microsoft Linear Algebra Subprograms ({ONNX} Runtime CPU kernels)},
  author       = {{Microsoft}},
  howpublished = {\url{https://github.com/microsoft/onnxruntime}},
  note         = {Accessed 2026}
}
@manual{armisa,
  title        = {Arm Architecture Reference Manual (ARMv8.2-A DotProd: SDOT/UDOT)},
  author       = {{Arm Ltd.}},
  note         = {ISA primary source}
}
@manual{intelisa,
  title        = {Intel Architecture Instruction Set Extensions Programming Reference (AVX-512 VNNI)},
  author       = {{Intel Corporation}},
  note         = {ISA primary source}
}

% ---- Edge / mobile inference benchmarking ----
@inproceedings{reddi2020mlperf,
  title     = {MLPerf Inference Benchmark},
  author    = {Reddi, Vijay Janapa and Cheng, Christine and Kanter, David and Mattson, Peter and Schmuelling, Guenther and Wu, Carole-Jean and others},
  booktitle = {ACM/IEEE International Symposium on Computer Architecture (ISCA)},
  pages     = {446--459},
  year      = {2020},
  note      = {arXiv:1911.02549; DOI:10.1109/ISCA45697.2020.00045}
}
@inproceedings{banbury2021mlperftiny,
  title     = {MLPerf Tiny Benchmark},
  author    = {Banbury, Colby and Reddi, Vijay Janapa and Torelli, Peter and others},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2021},
  note      = {arXiv:2106.07597}
}
@misc{reddi2020mobile,
  title  = {MLPerf Mobile Inference Benchmark},
  author = {Janapa Reddi, Vijay and others},
  year   = {2020},
  note   = {arXiv:2012.02328}
}
@inproceedings{ignatov2018,
  title     = {AI Benchmark: Running Deep Neural Networks on Android Smartphones},
  author    = {Ignatov, Andrey and Timofte, Radu and Chou, William and Wang, Ke and Wu, Max and Hartley, Tim and Van Gool, Luc},
  booktitle = {ECCV Workshops},
  year      = {2018},
  note      = {arXiv:1810.01109}
}
@misc{ignatov2019,
  title  = {AI Benchmark: All About Deep Learning on Smartphones in 2019},
  author = {Ignatov, Andrey and Timofte, Radu and others},
  year   = {2019},
  note   = {arXiv:1910.06663}
}
@misc{edgedetection2024,
  title  = {A Comprehensive Evaluation of Deep Learning Object Detection Models on Heterogeneous Edge Devices},
  author = {Alqahtani, Daghash K. and Cheema, Muhammad Aamir and Rodriguez, Maria A. and Toosi, Adel N.},
  year   = {2024},
  note   = {arXiv:2409.16808}
}
@misc{millar2025,
  title  = {Benchmarking Ultra-Low-Power {$\mu$}NPUs},
  author = {Millar, Josh and Huang, Yushan and Sethi, Sarab and Haddadi, Hamed and Madhavapeddy, Anil},
  year   = {2025},
  note   = {arXiv:2503.22567}
}

% ---- Numerical reproducibility / determinism (headline C2) ----
@misc{chen2026integeralibi,
  title  = {The Integer Alibi: Localizing Cross-Kernel Divergence in INT8-Quantized {LLM} Inference},
  author = {Chen, Teng-Ruei},
  year   = {2026},
  note   = {arXiv:2608.13756; companion to arXiv:2608.11693. Concurrent work.}
}
@misc{chen2026deterministic,
  title  = {Deterministic {LLM} Inference Across {GPU} Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance},
  author = {Chen, Teng-Ruei},
  year   = {2026},
  note   = {arXiv:2609.00363. Concurrent work.}
}
@inproceedings{li2021mqbench,
  title     = {MQBench: Towards Reproducible and Deployable Model Quantization Benchmark},
  author    = {Li, Yuhang and Shen, Mingzhu and Ma, Jian and Ren, Yan and Zhao, Mingxin and Zhang, Qi and Gong, Ruihao and Yu, Fengwei and Yan, Junjie},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2021},
  note      = {arXiv:2111.03759}
}
@misc{masoudian2026,
  title  = {What We Observe as {LLM} Behavior Can Be a Side-effect of Inference Backend},
  author = {Masoudian, Shahed and Shafaei, Passant and Swain, Monorama and Schedl, Markus},
  year   = {2026},
  note   = {arXiv:2608.04714}
}
@misc{fpnonassoc2024,
  title  = {Impacts of Floating-Point Non-Associativity on Reproducibility for {HPC} and Deep Learning Applications},
  author = {Shanmugavelu, Sanjif and Taillefumier, Mathieu and Culver, Christopher and Hernandez, Oscar and Coletti, Mark and Sedova, Ada},
  year   = {2024},
  note   = {arXiv:2408.05148}
}
@misc{repdl2025,
  title  = {RepDL: Bit-level Reproducible Deep Learning Training and Inference},
  author = {Xie, Peichen and Zhang, Xian and Chen, Shuo},
  year   = {2025},
  note   = {arXiv:2510.09180 (originally drafted 2023)}
}

% ---- Transformer quantization ----
@inproceedings{xiao2022smoothquant,
  title     = {SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author    = {Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  note      = {arXiv:2211.10438}
}
@inproceedings{dettmers2022llmint8,
  title     = {{LLM.int8()}: 8-bit Matrix Multiplication for Transformers at Scale},
  author    = {Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  note      = {arXiv:2208.07339}
}
@inproceedings{bondarenko2021,
  title     = {Understanding and Overcoming the Challenges of Efficient Transformer Quantization},
  author    = {Bondarenko, Yelysei and Nagel, Markus and Blankevoort, Tijmen},
  booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2021},
  note      = {arXiv:2109.12948}
}
@inproceedings{frantar2022gptq,
  title     = {{GPTQ}: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author    = {Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  note      = {arXiv:2210.17323}
}
@inproceedings{lin2023awq,
  title     = {{AWQ}: Activation-aware Weight Quantization for {LLM} Compression and Acceleration},
  author    = {Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle = {Conference on Machine Learning and Systems (MLSys)},
  year      = {2024},
  note      = {arXiv:2306.00978}
}
@inproceedings{yuan2022ptq4vit,
  title     = {PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization},
  author    = {Yuan, Zhihang and Xue, Chenhao and Chen, Yiqi and Wu, Qiang and Sun, Guangyu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  note      = {arXiv:2111.12293}
}
@inproceedings{liu2021ptqvit,
  title     = {Post-Training Quantization for Vision Transformer},
  author    = {Liu, Zhenhua and Wang, Yunhe and Han, Kai and Ma, Siwei and Gao, Wen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021},
  note      = {arXiv:2106.14156}
}

% ---- Vendor NPU toolchains (C3) ----
@misc{dhahri2025quanttrim,
  title  = {Quant-Trim in Practice: Improved Cross-Platform Low-Bit Deployment on Edge NPUs},
  author = {Dhahri, Rayen and Urban, Steffen},
  year   = {2025},
  note   = {arXiv:2511.15300; accepted to a EurIPS 2025 workshop (work in progress)}
}
@misc{qualcomm_qnn,
  title        = {Qualcomm AI Engine Direct ({QNN}) and {AI} Hub},
  author       = {{Qualcomm}},
  howpublished = {\url{https://app.aihub.qualcomm.com}},
  note         = {Vendor documentation; accessed 2026}
}
@misc{coreml,
  title        = {Core ML Tools ({coremltools})},
  author       = {{Apple}},
  howpublished = {\url{https://apple.github.io/coremltools}},
  note         = {Vendor documentation; accessed 2026}
}
@misc{litert,
  title        = {LiteRT (TensorFlow Lite) Delegates and NNAPI},
  author       = {{Google}},
  howpublished = {\url{https://ai.google.dev/edge/litert}},
  note         = {Vendor documentation; accessed 2026}
}

% ---- Accelerator characterization: roofline / data movement (C4) ----
@article{williams2009roofline,
  title   = {Roofline: An Insightful Visual Performance Model for Multicore Architectures},
  author  = {Williams, Samuel and Waterman, Andrew and Patterson, David},
  journal = {Communications of the ACM},
  volume  = {52},
  number  = {4},
  pages   = {65--76},
  year    = {2009},
  note    = {DOI:10.1145/1498765.1498785}
}
@inproceedings{chen2016eyeriss,
  title     = {Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks},
  author    = {Chen, Yu-Hsin and Emer, Joel and Sze, Vivienne},
  booktitle = {ACM/IEEE International Symposium on Computer Architecture (ISCA)},
  year      = {2016},
  note      = {DOI:10.1145/3007787.3001177}
}
@article{sze2017efficient,
  title   = {Efficient Processing of Deep Neural Networks: A Tutorial and Survey},
  author  = {Sze, Vivienne and Chen, Yu-Hsin and Yang, Tien-Ju and Emer, Joel S.},
  journal = {Proceedings of the IEEE},
  volume  = {105},
  number  = {12},
  pages   = {2295--2329},
  year    = {2017},
  note    = {arXiv:1703.09039; DOI:10.1109/JPROC.2017.2761740}
}

% ---- NVDLA / DLA & fixed-function INT8 accelerators ----
@misc{nvdla,
  title        = {{NVDLA}: {NVIDIA} Deep Learning Accelerator (Open Architecture)},
  author       = {{NVIDIA}},
  howpublished = {\url{http://nvdla.org}},
  note         = {Accessed 2026}
}
@inproceedings{farshchi2019nvdla,
  title     = {Integrating {NVIDIA} Deep Learning Accelerator ({NVDLA}) with {RISC-V} SoC on FireSim},
  author    = {Farshchi, Farzad and Huang, Qijing and Yun, Heechul},
  booktitle = {2nd Workshop on Energy Efficient Machine Learning and Cognitive Computing for Embedded Applications (EMC2)},
  year      = {2019},
  note      = {arXiv:1903.06495}
}
@misc{jetsonconcurrent2025,
  title  = {Profiling Concurrent Vision Inference Workloads on {NVIDIA} Jetson --- Extended},
  author = {Chakraborty, Abhinaba and Tavernier, Wouter and Kourtis, Akis and Pickavet, Mario and Oikonomakis, Andreas and Colle, Didier},
  year   = {2025},
  note   = {arXiv:2508.08430}
}

% ---- Automotive compute & redundancy (framing) ----
@manual{iso26262,
  title  = {ISO 26262: Road Vehicles --- Functional Safety},
  author = {{International Organization for Standardization}},
  year   = {2018}
}
@misc{nvidiadrive,
  title  = {{NVIDIA} DRIVE Functional-Safety Architecture},
  author = {{NVIDIA}},
  year   = {2018},
  note   = {Whitepaper}
}
@misc{hetsched2022,
  title  = {HetSched: Quality-of-Mission Aware Scheduling for Autonomous Vehicle SoCs},
  author = {Amarnath, Aporva and Pal, Subhankar and Kassa, Hiwot and Vega, Augusto and Buyuktosunoglu, Alper and Franke, Hubertus and Wellman, John-David and Dreslinski, Ronald and Bose, Pradip},
  year   = {2022},
  note   = {arXiv:2203.13396}
}
```
