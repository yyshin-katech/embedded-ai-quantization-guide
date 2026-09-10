# Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on Embedded and Automotive Accelerators

> **DRAFT — v0.3 (arXiv target).** Empirical sections wired from the repository's SSOT (32 measurement reports in `logs/`, 9 experiment suites in `experiments/`). Related Work (§2) and References are complete: all 46 BibTeX entries were verified against their arXiv/publisher pages (titles, authors, venues) on 2026-09-04. All numbers are relative comparisons under the caveats in §10. No internal infrastructure identifiers are included.

**Author:** Yuyeong Shin, Korea Automotive Technology Institute (KATECH)
**Contact:** yyshin@katech.re.kr
**Status:** pre-submission draft · target: arXiv (cs.LG / cs.PF / cs.AR)

---

## Abstract

Eight-bit integer (INT8) post-training quantization is the default recipe for edge deployment, under a widely held assumption: INT8 makes inference faster at a small, predictable accuracy cost, and a model quantized once can be carried to any target. We test that assumption with a controlled measurement study across seven hardware classes — ARM and x86 CPUs, a discrete GPU, an NVIDIA Jetson AGX Orin iGPU and its NVDLA cores, and two vendor NPUs (Qualcomm Hexagon HTP, DEEPX DX-M1) — holding the ONNX artifact and the quantization scales fixed so the integer kernel or ISA is the only free variable. Portability fails on three axes. **(1)** The *sign* of the INT8 speedup is set by the CPU's dot-product ISA (ARM `dotprod`/SDOT, x86 VNNI): cores that have it speed up by up to 2.1×, cores that lack it slow *down* by 1.7×, for the identical model and runtime. **(2)** INT8 outputs are not portable, and the rule is an invariance rather than a gradient: FP32 predictions are bit-identical for every pair (1000/1000), while INT8 predictions agree 1000/1000 exactly when two targets share an integer kernel and 958–965/1000 whenever they do not — independent of whether the boundary is CPU↔CPU or CPU↔accelerator, and invisible to top-1 accuracy, which is preserved. **(3)** Vendor NPUs own quantization: a bring-your-own QDQ graph fails silently on one NPU (external scales ignored, accuracy 0.75 → 0.005 while it compiles, profiles and runs without error) and loudly on the other (the compiler refuses the graph), so only the vendor's native path yields a correct engine. We further show that edge-NPU latency *regimes* are set by output/device-to-host transfer size rather than compute, and locate the transition with a fixed-compute sweep. We release the scripts and 32 reports. "Quantize once, deploy anywhere" is unsafe for embedded and automotive deployment, where per-input determinism and redundancy matter.

---

## 1. Introduction

Quantizing a floating-point network to 8-bit integers is the first and often only compression step in an embedded deployment pipeline. The operational folklore is compact and appealing: *(i)* INT8 is faster than FP32/FP16 because integer math is cheaper and moves less memory; *(ii)* the accuracy cost is small and bounded; and *(iii)* a model quantized once is a portable artifact — an ONNX file with QDQ nodes, or a TFLite model — that can be handed to any runtime or accelerator. This folklore underlies the way quantization is taught, benchmarked (report one speedup and one accuracy number), and shipped.

This paper asks whether the folklore survives contact with real, heterogeneous hardware. We ran a controlled measurement campaign across seven hardware classes and three model families (image classification, CNN object detection, and transformer detection), taking care to hold the model artifact and — critically — the quantization *scales* fixed across targets so that any difference we observe is attributable to the target's integer kernel or instruction set, not to a different quantizer.

We find that all three parts of the folklore fail, and they fail in ways that matter specifically for embedded and automotive systems:

1. **The speedup can be negative, and its sign is an ISA property (§4).** For the identical INT8 model on the identical CPU runtime (ONNX Runtime, MLAS), a Raspberry Pi 5 Cortex-A76 and a Jetson A78AE — which have ARM dot-product instructions — get 1.83× and 2.11× *faster*, while an x86 Core i9 without AVX-512 VNNI and a Cortex-A53 without dot-product get 1.65–1.76× *slower*. INT8 is not "faster"; it is faster *iff* the target has the right accumulate instruction.

2. **INT8 outputs are not numerically portable (§5, headline).** FP32 predictions are bit-identical across every platform pair where FP32 was measured (1000/1000). INT8 predictions are not: they disagree on ~4% of inputs across a CPU↔CPU pair and across the CPU↔GPU boundary (built from the *same* QDQ scales), and the rule is an invariance rather than a gradient: agreement is 1000/1000 whenever two targets share an integer kernel and 958–965/1000 whenever they do not, independent of whether the boundary is CPU↔CPU or CPU↔accelerator. A CPU↔vendor-NPU pair diverges further (939/1000), but its scales cannot be held fixed, so we report it as a deployment observation outside the invariance claim. Top-1 accuracy is preserved — the flips are net-neutral — so this is invisible to a standard accuracy report, yet it means a quantized model is not a deterministic function of its input once you change the target. Concurrent work localizes the underlying epilogue-rounding mechanism by swapping INT8 kernels on a *single* GPU for LLMs [chen2026integeralibi]; our contribution is orthogonal and, for deployment, more consequential — the cross-*physical-device* measurement for vision/detection models under an identical ONNX file and identical scales, with FP32 bit-identity as a control (§5).

3. **Vendors own quantization (§6).** A model you quantized yourself is not deployable to a vendor NPU as-is. Qualcomm's Hexagon HTP *silently* ignores external QDQ scales and collapses accuracy (0.75 → 0.005) while compiling, profiling, and running without error; the DEEPX compiler *loudly* rejects the same class of graph. Both are correct only through the vendor's own quantization path. These are opposite symptoms of one fact: the accelerator, not your toolchain, defines the numerics.

Beyond portability, we contribute a systems observation that reframes edge-NPU performance analysis: **latency regime is set by output/data-movement size, not compute (§7).** On one M.2 NPU, a classifier with a 4 KB output is compute-bound and scales 2.19× across cores, while a detector with a 2.82 MB raw output on the same device, same runtime, is data-movement-bound and does not scale at all — and a *lighter*-compute detector with an even larger output likewise refuses to scale (1.02×) and runs 26× slower than the classifier. A controlled single-variable sweep — compute held fixed, output swept 1020× — then traces the full transition curve and shows the regime boundary is not a knife-edge but a band whose width equals the core count (*N* cores share one data-movement link), yielding a closed-form rule for the output size at which adding cores stops helping. We isolate output size as the causal variable.

We frame these findings for their intended audience. The submitting institution works on automotive edge AI, where the failures above are not academic: non-portable INT8 numerics undermine the determinism and cross-module consistency that safety cases and redundant (dual-compute) architectures rely on. §8 and §9 add supporting characterization (transformer INT8 breakdown, DLA behavior) and a catalog of silent-failure pitfalls we hit, and §10 states the study's limits honestly.

**Contributions.**
- A controlled, same-artifact/same-scale cross-platform methodology that isolates the integer kernel/ISA as the only variable (§3).
- C1: the INT8 speedup sign is determined by the CPU dot-product ISA, shown across four CPUs (§4).
- C2 (headline): INT8 numerical non-portability across CPU↔CPU and CPU↔accelerator boundaries, with FP32 as a bit-identical control (§5).
- C3: vendor NPUs own quantization; two opposite BYO-QDQ failure modes (§6).
- C4: edge-NPU bottleneck regimes are set by output/data-movement size, not compute; a controlled fixed-compute sweep traces the transition, whose boundary is a band of width = core count (§7).
- A reproducibility artifact: scripts plus 32 measurement reports, and a claim-to-artifact map (Appendix A).

---

## 2. Background and Related Work

**Quantization foundations.** Integer-arithmetic-only inference — mapping weights and activations to INT8, accumulating products in INT32, and re-scaling back — was formalized by Jacob et al. [jacob2018] and codified for post-training use in the whitepapers of Krishnamoorthi [krishnamoorthi2018] and Nagel et al. [nagel2021whitepaper], with a broad survey by Gholami et al. [gholami2021survey]. The per-tensor vs. per-channel and symmetric vs. asymmetric design axes, and the result that symmetric INT8 typically stays within ~1% of FP32 for CNNs, are established by NVIDIA's evaluation [wu2020]; rounding choice at quantization time is itself accuracy-critical [nagel2020adaround]. Our work proposes no new quantization algorithm — it measures how *existing*, correctly-produced INT8 behaves once it crosses a hardware boundary. Crucially, Jacob's pipeline makes the INT32 accumulate exact and order-independent; the re-scaling *epilogue* is where implementation freedom — and, as we and concurrent work show, non-portability — lives.

**Integer kernels and dot-product ISAs.** The INT8 speed advantage depends on hardware dot-product/accumulate instructions: ARM `dotprod` (SDOT/UDOT, ARMv8.2-A) [armisa] and x86 AVX-512 VNNI [intelisa], exploited by the low-precision GEMM libraries gemmlowp [gemmlowp], XNNPACK [xnnpack], FBGEMM [khudia2021fbgemm], and Microsoft's MLAS [mlas] (the kernel library our ONNX Runtime CPU path uses). Prior characterization of data-center INT8 inference [park2018facebook] documents this dependence for throughput. We contribute the *cross-device sign-flip* framing — that the same model and runtime is faster or slower depending only on whether the target CPU has these instructions — and connect it to the accuracy-side consequence (§4 → §5).

**Edge and mobile inference benchmarking.** MLPerf Inference [reddi2020mlperf], MLPerf Tiny [banbury2021mlperftiny], MLPerf Mobile [reddi2020mobile], and AI-Benchmark [ignatov2018, ignatov2019] are the standard cross-device benchmarks, and recent studies benchmark detectors across the exact hardware family we use (Jetson, Raspberry Pi 5, Coral) [edgedetection2024, millar2025]. By design these quantize *per submission and per backend* and report per-device throughput/accuracy scores; none holds the quantization scales fixed across targets, and none reports cross-target numerical *agreement*. That axis — numerical portability under fixed scales — is our C2.

**Numerical reproducibility and determinism (closest to our headline).** The mechanism behind C2 was, concurrently with this work, localized by Chen [chen2026integeralibi, chen2026deterministic]: swapping only the INT8 GEMM kernel (CUTLASS vs. Triton) inside an LLM serving stack *on a single GPU* yields two engines that are each bit-reproducible against themselves yet agree on no generated sequence, with the divergence traced to scale application and output rounding in the *epilogue* (the INT32 accumulate being exact), and power-of-two scales restoring bit-identical cross-kernel agreement. We cite this as concurrent prior work and claim no discovery of the mechanism. Our contribution is orthogonal in setting and method: Chen swaps kernels on one device for LLMs, whereas we measure per-input prediction disagreement *across physical hardware boundaries* — dot-product CPU ↔ non-dot-product CPU, CPU ↔ GPU, and CPU ↔ vendor NPU — for vision and detection models, using an identical ONNX file with identical embedded QDQ scales, and we contrast the INT8 divergence against FP32 bit-identity on those same devices. More broadly, MQBench [li2021mqbench] measures a hardware-deployability *accuracy gap* across backends but not per-input, bit-level disagreement under identical scales; the inference backend has been shown to confound even greedy-decoding LLM behavior [masoudian2026]; and floating-point non-associativity [fpnonassoc2024] and fixed-reduction-order remedies [repdl2025] frame the FP side — we scope our "FP32 bit-identical" claim to our observed, fixed-thread configuration accordingly (§5, §10). The divergence is folklore in practitioner issue trackers for quantized TFLite CPU-vs-NPU and cross-EP ONNX Runtime outputs, but to our knowledge has not been systematically measured across embedded and automotive accelerators.

**Transformer quantization.** Transformer INT8 fragility is driven by activation outliers [dettmers2022llmint8, bondarenko2021], addressed by activation migration (SmoothQuant [xiao2022smoothquant]), activation-aware weight scaling (AWQ [lin2023awq]), and weight-only PTQ (GPTQ [frantar2022gptq]); for vision transformers specifically, PTQ4ViT [yuan2022ptq4vit] and Liu et al. [liu2021ptqvit] handle post-softmax/GELU activations. We use these to explain *why* DETR INT8 collapses on-device and to quantify how much the activation-granularity lever recovers on a real toolchain vs. in fake-quant (§8) — reinforcing that activations, not op selection, are the fragile axis.

**Vendor NPU toolchains.** Vendor compilers differ in scaling, clipping, and kernel support, so the same checkpoint yields inconsistent cross-backend accuracy — a point made by MQBench [li2021mqbench] and, most directly, by Quant-Trim [dhahri2025quanttrim], which proposes a training-time hardware-neutral checkpoint. Qualcomm's QNN/AI-Hub stack quantizes to its own native format [qualcomm_qnn], as do Apple Core ML [coreml] and TFLite/LiteRT delegates [litert]. These works establish that vendors prefer their own quantization; we contribute a controlled, cross-vendor account of *bring-your-own-QDQ rejection* and its two opposite failure modes — silent (accuracy collapse while running) vs. loud (compile refusal) — and show the native path is both correct and faster (§6).

**Accelerator characterization.** The compute-bound vs. memory-bound dichotomy is the Roofline model [williams2009roofline]; the primacy of data movement over compute energy is the Eyeriss line of work [chen2016eyeriss, sze2017efficient]. We extend the deployment-level picture with a third, output-size-driven regime — device-to-host (D2H)/PCIe-bound — and show that on one PCIe-attached edge NPU the bottleneck (and whether multi-core helps) is set by the model's output tensor size, not its compute (§7). Concurrent-inference profiling on Jetson [jetsonconcurrent2025] corroborates our related finding that GPU-fallback subgraphs serialize otherwise-parallel accelerator work.

**NVDLA and fixed-function INT8 accelerators.** The Orin DLA is an NVDLA v2 instance [nvdla, farshchi2019nvdla]; we characterize it as an INT8-only, CNN-favoring datapath that is the perf-per-watt leader for CNNs but fragments on transformers (§8).

**Automotive compute and redundancy (framing).** Redundant, diverse compute across CPU/GPU/DLA is the backbone of automotive functional-safety architectures (ISO 26262 / ASIL [iso26262], NVIDIA DRIVE [nvidiadrive]) and heterogeneous AV-SoC scheduling [hetsched2022]. This is our motivation: non-portable INT8 numerics (§5) directly threaten the cross-module agreement that dual-compute redundancy assumes — the concern that motivates our planned follow-on work on a multi-module platform (§11).

---

## 3. Experimental Methodology

### 3.1 Hardware matrix

The table below lists the eight targets, grouped into the seven hardware classes this study spans, and the role each one plays.

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

Every device is a single unit (n=1 per class); we therefore make *relative* claims about representative devices, not population claims about an ISA (see §10).

### 3.2 Models and datasets

ResNet-18/50 (ImageNet-1k classification), DETR-ResNet-50 (COCO detection, transformer), YOLO26n and YOLOv5s (COCO detection, CNN), plus BEVFormer/BEVDet (3D BEV) used only for latency/engine characterization. Accuracy is reported on ImageNet val (subset or full, stated per result) and COCO val2017 subsets. Absolute accuracy/latency are not comparable across sections because batch size, input resolution, and evaluation subset differ; we report *relative* deltas within a controlled comparison.

### 3.3 Controlled-comparison principle

The core methodological device of this paper: for any cross-target comparison, we fix the model artifact and the quantization *scales*, so the only free variable is the target's integer kernel/ISA. Concretely, the same `resnet50_int8_qdq.onnx` (with its embedded QDQ scales) is (a) run on multiple CPU EPs and (b) used to *build* the TensorRT INT8 engine — so the GPU integer kernel and the CPU integer kernel consume identical scales. When a comparison cannot hold scales fixed (e.g., a vendor NPU that rejects external QDQ), we say so and treat the result as a deployment finding, not a kernel comparison.

### 3.4 Measurement protocol and scope

Latencies are event-timed on GPU/accelerator paths and wall-clock on CPU/harness paths (the two are not directly comparable and are never mixed within a claim). Unless noted, batch size is 1. Numerical agreement is reported as the number of inputs (out of a fixed bundle, typically 1000 for classification) on which two targets produce the *same* top-1 prediction. **Known limitation:** most latencies are single-run p50 without confidence intervals, and several accuracy numbers are on subsets; §10 quantifies why this bounds our claims to relative comparisons, and this study's own §9 shows subset evaluation can inflate top-1 by ~9.77 percentage points.

---

## 4. The Speedup Sign Is ISA-Determined (C1)

We ran the identical ResNet-50 INT8 QDQ model on the identical runtime (ONNX Runtime, CPU execution provider, MLAS kernels) on four CPUs, and compared against the same model in FP32 on each. The table below and Figure 1 report the four measurements.

| CPU | Dot-product ISA | FP32 → INT8 | INT8 effect |
|---|---|---|---|
| Cortex-A76 (Raspberry Pi 5) | ARM `dotprod` (SDOT) | 144.95 → 79.08 ms | **1.83× faster** |
| Cortex-A78AE (Jetson AGX Orin) | ARM `dotprod` | 38.47 → 18.22 ms | **2.11× faster** |
| Core i9-10900K (x86) | no AVX-512 VNNI | 9.28 → 16.34 ms | **1.76× slower** |
| Cortex-A53 (i.MX8M-Nano) | no dot-product | 680.20 → 1123.02 ms | **1.65× slower** |

**Figure 1.** The INT8 speedup sign is an ISA property. Bar length is the log-scaled speedup (FP32 latency ÷ INT8 latency) for the identical ResNet-50 INT8 QDQ model on the identical ONNX Runtime CPU (MLAS) path; bars to the right of 1× are faster under INT8, bars to the left are slower. The two cores with an ARM dot-product instruction gain; the two without one lose. Same data as the §4 table.

*(Rendered in the LaTeX build as a TikZ chart; the data is the table above.)*

The determinant is not the ISA *family* (both ARM and x86 appear on both sides): the A53 is ARM yet slows down, the A76/A78AE are ARM and speed up. The determinant is the presence of a dot-product/accumulate instruction (ARM `dotprod`, x86 VNNI). Without it, INT8's re-quantization and widening overhead is not amortized by a faster inner product, and INT8 is a *pessimization*. This has an immediate practical consequence: a fleet of heterogeneous edge CPUs cannot assume INT8 is a win; the same binary regresses on the wrong core.

---

## 5. INT8 Outputs Are Not Portable (C2) — headline

If two targets run the *same* quantized model with the *same* scales, do they produce the same predictions? For FP32 the answer is yes, exactly. For INT8 it is no.

**FP32 is a bit-identical control.** Across every platform pair where it was measured, FP32 top-1 predictions agree on 1000/1000 inputs. Whatever divergence we see under INT8 is therefore not a floating-point reduction-order artifact of our harness; it is specific to the integer path. We scope this bit-identity to our observed configuration (fixed thread count, a single reduction path per target): floating-point non-associativity can make FP reductions non-deterministic under different parallelization or hardware [fpnonassoc2024], so the precise claim is "FP32 was bit-identical across these targets as measured," not that FP32 is portable by construction. The contrast we rely on — FP32 identical, INT8 diverging, on the *same* devices and harness — holds regardless.

**INT8 disagreements, ordered by kernel divergence.** The table below lists every pair we measured; Figure 2 plots the same data on a single agreement axis, where the bimodality is immediate.

| Comparison | Boundary | Integer kernel | FP32 | INT8 |
|---|---|---|---|---|
| Jetson A78AE vs. Pi 5 A76 | CPU↔CPU | **same** (MLAS SDOT) | 1000 / 1000 | **1000 / 1000** |
| i.MX8M-Nano A53 vs. Pi 5 A76 | CPU↔CPU | different | 1000 / 1000 | 965 / 1000 |
| i.MX8M-Nano A53 vs. x86 i9 | CPU↔CPU | different | 1000 / 1000 | 961 / 1000 |
| Raspberry Pi 5 A76 vs. x86 i9 | CPU↔CPU | different | 1000 / 1000 | 958 / 1000 |
| Jetson iGPU (TensorRT) vs. A78AE (MLAS) | **CPU↔accelerator** | different | 1000 / 1000 | **961 / 1000** |
| DEEPX DX-M1 (NPU) vs. host A76 CPU ‡ | **CPU↔vendor-NPU** | different, *and different scales* | — | 939 / 1000 |

**Figure 2.** INT8 agreement is bimodal, and the location of the hardware boundary does not predict it. Each row is one target pair running the same model artifact: ○ is FP32 top-1 agreement, ● is INT8. FP32 is 1000/1000 wherever it was measured. INT8 is exactly 1000/1000 for the one pair that shares an integer kernel and falls in a 958–965/1000 band (shaded) for every pair that does not — including the CPU↔accelerator pair, which sits *inside* the CPU↔CPU range. The DEEPX row (‡, gray) cannot hold the scales fixed (§6) and is a deployment observation, excluded from the invariance claim. Same data as the §5 table.

*(Rendered in the LaTeX build as a TikZ chart; the data is the table above.)*

**The result is an invariance, not a gradient.** Across the five pairs that share one model artifact and its embedded scales, agreement takes essentially two values: **1000/1000 when the two targets execute the same integer kernel, and 958–965/1000 when they do not.** Nothing else predicts it — in particular, the *location* of the hardware boundary does not. The CPU↔accelerator pair (961/1000) falls inside the range spanned by the three CPU↔CPU pairs (958–965/1000), and the one pair that agrees perfectly is itself a CPU↔CPU pair spanning two different SoCs, two different boards, and two ONNX Runtime versions (1.23.2 and 1.28.0). Sharing the integer kernel is *sufficient* for identical predictions; crossing a chip boundary is *not sufficient* to break them. What breaks them is a different re-quantization epilogue.

The CPU↔GPU row makes the mechanism explicit: it uses the *same* `resnet50_int8_qdq.onnx` to build the TensorRT engine that the CPU EP runs, so the QDQ scales are bit-identical and the only free variable is the integer kernel (TensorRT's INT8 kernels vs. MLAS's) — yet the two disagree on 39/1000 inputs. The divergence is kernel-specific rounding and accumulation, not hardware noise or a scale mismatch.

‡ **The vendor-NPU row is excluded from the invariance claim.** It is the one comparison here that cannot hold scales fixed: the DEEPX compiler rejects an external QDQ graph outright (§6) and runs its own PTQ, so the pair differs simultaneously in integer kernel, quantizer, and calibration set. Per the controlled-comparison principle of §3.3 we therefore report 939/1000 as a *deployment* observation — what a fielded CPU-plus-vendor-NPU pairing actually delivers — and not as a kernel comparison. It is consistent with, but does not evidence, the invariance above.

**Accuracy hides it.** The disagreements are net-neutral: on the Jetson iGPU, accuracy-valid INT8 top-1 is 0.7620 — *lossless* versus its own FP32 and higher than the CPU MLAS INT8 top-1 of 0.7500 — even though the two INT8 paths flip predictions on dozens of individual inputs. A standard "accuracy after quantization" report would show no problem. The portability failure is only visible when you compare *predictions per input across targets*.

**Relation to concurrent work.** The mechanism behind these disagreements — divergence introduced in the re-quantization *epilogue* (scale application and output rounding) after an exact INT32 accumulate — was localized concurrently by Chen [chen2026integeralibi, chen2026deterministic], who swapped INT8 GEMM kernels (CUTLASS vs. Triton) *on a single GPU* for LLMs and showed that power-of-two scales restore bit-identical cross-kernel agreement. We do not claim to discover this mechanism, and we cite that work as concurrent prior art. Our result is complementary and, for deployment, more consequential in three ways: the divergence persists across *different physical devices* (a dot-product CPU vs. a non-dot-product CPU, a CPU vs. a GPU built from the same scales, a CPU vs. a third-party vendor NPU); it holds for vision and detection models rather than LLMs; and it is measured against an FP32 bit-identical control on the same devices. Where Chen asks whether two kernels on one GPU agree, we ask whether a fielded fleet of heterogeneous targets agrees — the question a cross-module automotive platform actually poses.

**We tested that mitigation across a physical boundary; it does not transfer.** Chen's power-of-two result is demonstrated for two kernels on one GPU. We forced *every* Q/DQ scale in `resnet50_int8_qdq.onnx` to a power of two — making the requant multiplier `M = (s_a·s_w)/s_out` a pure shift `2^(k_a+k_w−k_out)` that any kernel *should* implement identically — rebuilt the model (weights re-quantized to the new scales; the zero-points are already all zero, so `M` is the only divergence term), and re-ran the x86 ↔ Pi 5 (A76) pair. It fails, and informatively: agreement does not rise to 1000/1000 but *drops below the baseline*, 958/1000 → **869/1000** (ceil rounding, scales raised by ≤1 bit) and **919/1000** (nearest, ≤√2) — a gate FAIL on both arms. MLAS does not special-case `M = 2^k` into a shared exact shift; forcing powers of two instead coarsens the requantization grid and *enlarges* the epilogue divergence it was meant to remove — the per-element cross-kernel |Δlogit| grows ×2.35 (nearest) to ×4.34 (ceil), monotonically with the flip count. A mechanism probe (report `stage5_pot_scales_report.html`) separates the two possible causes: the divergence-magnitude increase is confirmed and amplified, while the competing tie-breaking hypothesis — that power-of-two scales create more exact `.5` ties for half-even vs. half-away kernels to split — is *refuted*, the near-tie population staying essentially flat (41 → 36 / 45 across baseline / ceil / nearest) and moving *opposite* to the flip count. All figures are exact rather than point estimates: 30 repeat runs are bit-identical. Chen's single-GPU finding is not contradicted; our measurement shows only that it does not carry across the x86↔A76 boundary — the power-of-two mitigation whose scope §11 bounds.

**Why it matters.** A quantized model is often treated as a deterministic function `f(x)`. C2 says that once you change the target, `f` changes on a measurable fraction of inputs, silently, with no accuracy signal. For automotive systems this is the crux: dual-compute redundancy and cross-module consistency checks assume two units computing the same input agree; C2 shows that assumption holds under FP32 but not under INT8 across heterogeneous integer kernels. The invariance sharpens what the design constraint is: determinism does not degrade gradually with hardware distance, so it cannot be bought back by choosing a "closer" second target. It is binary in the integer kernel. Two units agree per-input only if they run the same integer kernel — which, across a heterogeneous redundant architecture, is precisely the property the architecture was chosen to *avoid*.

---

## 6. Vendors Own Quantization: Two Failure Modes (C3)

C1–C2 assume you can even *run* your quantized model on the target. On vendor NPUs, you frequently cannot — the accelerator insists on quantizing the model itself.

**Qualcomm Hexagon HTP — silent.** Submitting an externally quantized ONNX-QDQ ResNet-50 to Qualcomm AI Hub compiles, profiles, and runs on-device with no error, but the HTP ignores the external QDQ scales and on-device top-1 collapses from 0.75 to **0.005**. The same ONNX runs correctly on x86 CPU (0.753), and the FP32/fp16 path on HTP is faithful (0.745) — so the failure is specific to *externally supplied* INT8 scales being discarded. The correct path is the vendor's own `submit_quantize_job` (HTP-native QDQ), which recovers top-1 to **0.735** and is *faster and leaner* than the external-QDQ engine (748 µs vs. 1052 µs).

**DEEPX DX-M1 — loud.** Feeding the same class of externally quantized graph to the DEEPX compiler produces a hard error — `GraphStructureError: 106 isolated node(s)` → `InternalError`, with no engine emitted. The native path (supply FP32; let the DEEPX compiler run its own PTQ) compiles cleanly and reaches top-1 **0.7660**, lossless-grade.

**One root cause, opposite symptoms.** Qualcomm fails *open* (runs a silently broken model — dangerous, because a broken model can ship) and DEEPX fails *closed* (refuses to build — safe, because nothing broken can ship). Both encode the same rule: the accelerator, not your toolchain, owns the quantization. The deployment consequence is that a quantization you validated on one target is not a portable artifact to a vendor NPU at all — reinforcing C2 from the deployability side, and adding a concrete safety-relevant hazard in the Qualcomm case (a silently wrong INT8 model that passes compile and profile).

---

## 7. Bottleneck Regimes Are Set by Output Size, Not Hardware (C4)

Reasoning about "is this NPU fast enough" usually starts from compute (FLOPs/TOPS). On a PCIe-attached edge NPU we find the latency *regime* — and whether adding cores helps at all — is set by the model's output (device-to-host, D2H) transfer size, on the same device and runtime. The table below puts three models on that one accelerator.

| Model (on DEEPX DX-M1) | Output size | Regime | Multi-core scaling |
|---|---|---|---|
| ResNet-50 | 4 KB | **compute-bound** (compute 2.77 ms ≫ D2H 0.11 ms) | 2.19× near-linear on 3 cores |
| YOLO26n | 2.82 MB (raw head) | **D2H-bound** (D2H 21.81 ms ≫ compute 9.0 ms) | 1.00× flat |
| YOLOv5s | 5.48 MB | **D2H-bound**, worse | 1.02× flat; lightest compute (2.59 ms) yet **26× slower than ResNet-50** |

The YOLOv5s row isolates the causal variable: it has the *smallest* compute of the three yet is by far the slowest — 41.01 fps on three cores against ResNet-50's 1078.93 fps, a 26.3× gap in the wrong direction — because it has the largest output to move across the PCIe Gen2×1 link. Compute does not predict the regime; output/D2H size does.

**A controlled single-variable sweep traces the transition curve.** The three models above bracket the two regimes but cannot trace the boundary between them, because they differ in compute as well as output. We remove that residual confound with a synthetic sweep on the same device and runtime: a fixed heavy convolutional trunk holds NPU compute constant (1-core inference p50 = 2.70 ms, spread 4.9% across the sweep) while a 4-channel bottleneck feeding a 1×1 "expander" head varies *only* the output tensor, over 1020× (3,920 B → 3,999,968 B; 12 points). Holding compute fixed and sweeping output alone, 3-core throughput scaling sits on a flat compute-bound plateau of 2.97–2.99× while the output stays below ≈63 KB, then descends monotonically to 1.00× (D2H-bound), and the per-core job distribution migrates 33/33/33 → 98/2/0 as the one shared PCIe link progressively starves all but one core (reproducing the YOLO26n 472/28/2 signature). The single-inference crossover — where D2H equals the fixed 2.70 ms compute — falls at ≈1.05 MB of output. The table below lists five representative points of the sweep; Figure 3 plots all twelve.

| Output bytes (fixed 2.70 ms compute) | D2H p50 | 3-core scaling |
|---|---|---|
| 3,920 | 0.141 ms | 2.98× (compute-bound plateau) |
| 256,368 | 0.776 ms | 1.99× (transition band) |
| 511,952 | 1.367 ms | 1.59× (transition band) |
| 1,000,384 | 2.429 ms | 1.21× (transition band) |
| 1,999,984 | 8.892 ms | 1.00× (D2H-bound) |

*(Output sizes are exact byte counts, not rounded KB/MB, because the two regime thresholds below fall between conventional round numbers.)*

**Figure 3.** The compute-bound → D2H-bound transition with NPU compute held fixed. All 12 synthetic sweep points on the DEEPX DX-M1: 3-core throughput scaling against output (D2H) size, at a constant 2.70 ms 1-core inference (spread 4.9%). Scaling holds a 2.97–2.99× plateau while the output stays below ≈63 KB, then descends to 1.00×. The shaded band runs from the link-saturation onset (326,377 B, where D2H = compute/*N*) to the single-inference crossover (1,095,949 B, where D2H = compute); its 3.36× width equals the core count *N* = 3, because *N* cores share one D2H link. Five of these points are listed numerically in the §7 sweep table.

*(Rendered in the LaTeX build as a TikZ chart; the data is the table above.)*

**The regime boundary is a band whose width equals the core count.** The transition is not a knife-edge but a band bounded by two thresholds: multi-core scaling begins to break when D2H reaches compute/*N* (≈319 KB — the one shared link can no longer feed all *N* cores), and a *single* inference turns D2H-bound when D2H reaches compute (≈1.05 MB). Their ratio is *N*: we measure a 3.36× band on a 3-core device, because *N* cores share one D2H link, so aggregate throughput saturates at 1/*N* of the per-core D2H that bounds one inference. This gives a closed-form provisioning rule — from compute time and link bandwidth alone, an *N*-core accelerator behind a bus stops scaling once the output exceeds ≈(compute-time × link-bandwidth)/*N* — turning the qualitative "provision by data movement" into a quantitative threshold. (Thresholds are linear-fit extrapolations, 2.456 ms/MB, specific to this Gen2×1 link; the band-width = *N* ratio is bandwidth-independent. See §10.)

We observe a third regime with the transformer detector DETR, where the DEEPX compiler auto-splits the graph and leaves the transformer on the host CPU in FP32: end-to-end 1036.34 ms decomposes as host-CPU transformer FP32 910.6 ms (87.9%) ≫ D2H 57.28 ms ≫ NPU INT8 41.11 ms (4.0%) ≫ H2D 6.97 ms — **host-CPU-compute-bound**. (The four stages sum to 1015.96 ms, which matches the runtime's own end-to-end p50 of 1017.53 ms; the 1036.34 ms figure is the benchmark harness's wall-clock, so the ≈20 ms excess is harness overhead rather than unaccounted device time.) Three models on one accelerator thus exhibit three different bottlenecks (NPU-compute, PCIe-D2H, host-CPU-compute). For context, in its favorable (compute-bound) regime the same NPU delivers large wins over the host CPU — e.g., YOLO26n throughput 91.51 fps vs. 8.01 fps on the A76 (×11.42) and host-side perf/watt ×29.29 — but those wins evaporate the moment the model's output pushes it into the D2H-bound regime. The design rule: for edge accelerators behind a bus, provision and partition by data movement, not by TOPS.

---

## 8. When INT8 Breaks Down: Transformers and the Granularity Lever (supporting)

C1–C3 use CNNs, where INT8 is (kernel permitting) nearly lossless. Transformers are the stress test and reinforce C2's thesis that *activations*, not ops, are where INT8 portability breaks.

### 8.1 The collapse, and what does *not* explain it

DETR INT8 collapses hard and reproducibly: FP32 mAP 0.4207 → INT8 0.2402 (−42.9%) on a discrete GPU (ORT `quantize_static`, QDQ format, per-channel `QInt8` weights and **per-tensor** `QInt8` activations, MinMax calibration over 100 images, CUDA EP, COCO val2017 5,000 images), cross-confirmed on Jetson with symmetric re-quantization (0.4237 → 0.2383, −43.8%), with small-object mAP down 77–85%.

**Op selection is not the lever.** Leaving all 36 attention-score matmuls in FP barely moves the result (0.2402 → 0.2438, **+0.0036 mAP**). Three of the four exclusion patterns routinely recommended for transformer PTQ are in fact no-ops on this graph: DETR contains no GELU, and ORT's QDQ quantizer does not touch Softmax or LayerNorm to begin with. Only the attention-matmul exclusion is even applicable, and it recovers 2% of the gap.

**Nor is the damage owned by one half of the network.** Quantizing each half alone shows both halves collapse on their own:

| Configuration (ORT, COCO val2017 5,000) | mAP | Δ vs. FP32 |
|---|---|---|
| FP32 | 0.4207 | — |
| Backbone only INT8 (53 Convs) | 0.2653 | **−36.9%** |
| Transformer only INT8 (137 nodes) | 0.2391 | −43.2% |
| All INT8 (190 nodes) | 0.2402 | −42.9% |

The two halves' losses are strongly **sub-additive** — either half alone reproduces most of the full collapse, and their separate losses sum to far more than the whole. The damage is therefore distributed across the network rather than localized to a fragile subgraph. What the two halves share is not an op type but the quantizer's *activation* treatment: one per-tensor MinMax scale per activation tensor. That is the axis §8.3 isolates.

The activation-granularity lever (SmoothQuant) recovers 59.9% of the gap in a torch fake-quant setting but only ~9% on-device — because the only on-device-buildable INT8 path quantizes Gemms only (attention/LayerNorm/Softmax stay FP16), so the lever cannot reach the dominant residual.

### 8.2 A toolchain that declines: no collapse by avoidance

The DEEPX compiler produces *no* transformer INT8 collapse on the same model (mAP 0.4377 → 0.4385, **+0.0008**, within noise). The reason is not better transformer quantization: the compiler auto-partitions the 708-node graph and assigns only the CNN backbone and the first encoder self-attention block to the NPU in INT8, leaving the remaining encoder layers, all six decoder layers, the FFNs and the heads on the **host CPU in FP32**. The handoff tensor is the giveaway — 23.99 MB of FP32 attention state crosses back to the host per frame.

"No collapse" and "collapse" are thus two sides of one fact: transformer activations do not survive INT8, so a toolchain either refuses (no loss, no speedup — §7 shows the resulting pipeline is host-CPU-bound at 1.04 s end-to-end) or forces it (speedup, large loss). This is the portability thesis again, in the accuracy dimension.

### 8.3 The quantizer does not port either

§8.2 leaves a matched pair we can exploit. Both toolchains were, in effect, asked to run the *same nominal recipe* — INT8 the CNN backbone, keep the transformer in FP32 — and both report an FP32 reference computed with the transformer in FP32. Their answers differ by 37.1 points of relative mAP.

| | ORT (§8.1, discrete GPU) | DEEPX `dx_com` 2.4.0 (DX-M1) |
|---|---|---|
| INT8 scope | backbone Conv ×53 (`input_projection` excluded) | backbone **+ `input_projection` + first encoder self-attn through its Softmax** |
| Weights | per-channel `QInt8` | `wbit8` |
| **Activations** | **per-tensor `QInt8`, MinMax, 100 images** | **`abit8`, EMA, 100 in-domain COCO images** |
| Input shape at calibration | dynamic axes (varies per image) | fixed 800×1066, bit-identical `.npy` |
| Evaluation | COCO val2017, 5,000 images | COCO val2017, first 500 images |
| FP32 → INT8 | 0.4207 → 0.2653 | 0.4377 → 0.4385 |
| **Relative Δ** | **−36.9%** | **+0.2% (within noise)** |

The obvious explanations do not survive contact with the rows. The INT8 *scope* runs the wrong way: DEEPX quantizes a strictly larger portion of the graph — including the input projection and an entire attention block that ORT leaves in floating point — and loses less. Evaluation subset (5,000 vs. 500) and fixed-vs-dynamic resolution shift a mAP by ones of percent, not by 37.1 points.

What remains is the activation calibrator. A per-tensor MinMax scale is set by the single largest activation observed during calibration, so one outlier image dilates the range and quantizes every other activation coarsely; an EMA calibrator damps exactly that. The ORT path amplifies the effect further by calibrating over dynamic input shapes, so the 100 calibration images do not even share a common activation geometry. This is the §8.1 lever — activation granularity and variance, not op selection — now observed *across two vendors' quantizers* rather than within one.

The deployment consequence is a fourth portability axis, and in practice the sharpest. C2 (§5) shows that holding the scales fixed still leaves the *outputs* target-dependent. §8.3 shows that when you *cannot* hold the scales fixed — which §6 establishes is the normal case on a vendor NPU — even the *accuracy* of a nominally identical recipe becomes a property of the vendor's calibrator rather than of your model. "We quantized the backbone and it was fine" is not a transferable statement.

*(Caveat.* The two columns are not comparable in absolute mAP: different evaluation subsets, resolutions, and runtimes. Only the within-column FP32→INT8 relative deltas are compared — which is precisely the quantity that differs by 37.1 points, and each column's FP32 reference is measured on its own pipeline.)*

*(Accelerator note.* On the Jetson NVDLA v2, INT8 is not merely preferred but mandatory: DLA is an INT8-only datapath (FP16 is 13.87× slower) and is the perf-per-watt leader (51.29 inf/s/W, ~1.55× the iGPU at roughly half the power) for CNNs, but fragments on transformers — the identical `--useDLACore=0 --allowGPUFallback` recipe that gives ResNet-50 a clean 2-fallback offload gives DETR 326 DLA layers, 404 GPU-fallback layers and 16 ForeignNodes, at 398.64 ms versus 13.28 ms for the same model in FP16 on the iGPU (**30× slower**, FP16 vs. FP16). The accelerator's "preferred precision" is itself a non-portable, model-dependent property.)*

---

## 9. Methodology Pitfalls and Silent Failures (C8)

The measurements above were only trustworthy after we removed a series of silent errors that a normal pipeline would not surface. We report them because they bound what any single-number quantization result means.

- **Subset evaluation inflates accuracy.** A 1000-image ImageNet subset overstated top-1 by **+9.77 percentage points on average** versus the full 50k val set — a mean over eight model configurations, spanning +8.41 to +10.39 pp, so the inflation cannot be corrected away by a constant offset without reordering the models. It also flipped the sign of three quantization deltas and 5 of 13 significance verdicts. Accuracy claims here are either on full val or explicitly flagged as relative-on-subset.
- **Preprocessing dominates the quantization delta.** The choice of resize/crop (squash vs. torchvision) changed top-1 by **−1.07 pp**, roughly **9×** the −0.12 pp cost of the quantization itself. A quantization number is meaningless without a fixed, reported preprocessing.
- **SQNR does not predict accuracy.** Per-layer SQNR had essentially no rank correlation with the top-1 delta (Spearman ρ = −0.030 over 21 layers, 50k val); the common practice of ranking layers by SQNR to guide mixed precision is not supported here.
- **Silent fallbacks are everywhere.** External QDQ silently falling back to CPU/FP32; a `TensorrtExecutionProvider` that is listed as available yet quietly runs on the CPU because `libnvinfer.so.10` was off the loader path (p50 11.83 ms, i.e. CPU-class, versus 0.41 ms once fixed); a zero-copy output buffer aliasing bug that collapsed top-1 to 0.0014 with no error; an opset down-convert that "succeeds" (exit 0) while producing an invalid graph; the pip TensorRT wheel shipping without the `trtexec` binary the tutorials assume. Each of these produces a plausible-looking number that is wrong.

---

## 10. Threats to Validity

We state the study's limits plainly; several are properties of a measurement-first project and bound our claims to *relative* comparisons.

- **Single-run latencies without confidence intervals.** Most latencies are p50 or single-run. We do not claim differences smaller than a few percent; the headline results (sign flips of 1.6–2.1×, regime differences of >20×) are far outside plausible run-to-run noise, but the smaller ones should be read as directional.
- **Subset accuracy.** Several accuracy and agreement numbers are on subsets (200–1000 images). This study itself quantifies the risk (§9, +9.77 pp mean inflation); the agreement counts (958–1000/1000) are on fixed 1000-image bundles and are internally comparable, but not comparable to full-val absolute accuracy.
- **n=1 per hardware class.** One unit per class. We claim "this representative device," never "all A76" or "all x86."
- **Version confounds, and the controls that bound them.** Runtime versions differ across targets and cannot be equalized: the four CPUs ran ONNX Runtime 1.17.1 (A53), 1.23.2 (A78AE and x86 i9) and 1.28.0 (Pi 5), so *every* cross-platform pair in C1 and C2 is also a cross-machine comparison, and three of the four CPU↔CPU pairs are cross-version as well. Two controls bound what the version difference can explain. (i) FP32 predictions are 1000/1000 across all three ORT versions, so version alone does not perturb predictions on this graph. (ii) The one INT8 pair that agrees perfectly (A78AE↔Pi 5, 1000/1000) is itself cross-version (1.23.2 vs. 1.28.0), while a pair that disagrees (A53↔x86, 961/1000) is cross-version too — version does not separate the two outcomes, the integer kernel does. For C1, the sign flip survives within a single version: A78AE and the x86 i9 both ran 1.23.2, and INT8 made one 2.11× faster and the other 1.76× slower. We nonetheless report the version of each target and treat any residual version effect as a limit on absolute latencies, not on the sign or the agreement invariance.
- **Relative, not absolute.** Batch size, input resolution, and evaluation subset differ across sections; absolute latency/accuracy are not cross-comparable. All claims are within-comparison relative deltas.
- **Power-measurement gap.** Some perf-per-watt figures use a host-side power boundary because on-board/M.2 card power (upstream of the accessible rail) or DLA power (not captured by the GPU utilization counter) could not be isolated; we report the measurement boundary alongside each figure.
- **Init-weight models excluded from accuracy.** The BEV capstone models ran with initialization weights (public weights unavailable), so their mAP is ~0 by construction and is used only for latency/engine-size characterization, never for accuracy claims. The §7 regime-transition sweep likewise uses synthetic random-weight models with a synthesized input; they are valid only for latency/regime, never accuracy.
- **Extrapolated, link-specific thresholds (§7).** The two transition thresholds (≈319 KB, ≈1.05 MB) are extrapolations of a linear D2H fit (2.456 ms/MB), not directly measured points, and their *absolute* positions are specific to this host's PCIe Gen2×1 link (a wider link shifts them). The structural result — a transition band of width = core count — is link-bandwidth-independent, since it follows only from *N* cores sharing one D2H link.
- **Vendor scope.** Vendor-NPU findings cover Qualcomm and DEEPX; other automotive NPUs (TI, Renesas) were not available and are left to future work.
- **Power-of-two mitigation — scope of the negative result (§5).** The finding that forcing power-of-two scales does not restore cross-device bit-identity — and in fact makes agreement worse — is measured on one model (`resnet50_int8_qdq.onnx`), two rounding modes (ceil, nearest), and one physical boundary (x86 no-VNNI ↔ A76 SDOT, both MLAS). It is an input/output measurement, not kernel introspection: the mechanism attribution (grid coarsening enlarges the epilogue divergence; tie-breaking does not) rests on the mutual consistency of the |Δlogit| magnitudes, the weight-saturation counts (ceil 0 vs. nearest 41,447), and the decision-margin distributions across the two rounding modes, not on reading the kernel source. It refutes the *transfer* of Chen's single-GPU result to this boundary; it does not refute that result in its own setting, and a different boundary (CPU↔accelerator, CPU↔vendor-NPU) or a kernel pair that *does* implement `M=2^k` as an exact shift could behave differently.

---

## 11. Conclusion

Across seven hardware classes we find that INT8 quantization is not portable on any of the three axes the operational folklore assumes. Its *speedup* can be negative and its sign is set by the CPU's dot-product ISA. Its *numerics* are not portable: identical scales produce disagreeing predictions across CPU↔CPU and CPU↔accelerator boundaries, while FP32 stays bit-identical — an accuracy-invisible loss of determinism. Its *deployability* is gated by the target vendor, which owns quantization and rejects a bring-your-own QDQ graph either silently (dangerously) or loudly. We add that edge-NPU latency regimes are governed by data movement, not compute.

The practical recommendations are concrete: re-validate INT8 per target rather than once; treat vendor-native quantization as mandatory, not optional; provision accelerators by output/data-movement size; and, for safety-relevant or redundant automotive compute, do not assume two heterogeneous units running the same INT8 model agree per input — they do under FP32 and may not under INT8. Where per-input cross-target determinism is required, constraining quantization to power-of-two scales — shown to restore bit-identical cross-kernel agreement in the single-GPU LLM setting [chen2026deterministic] — is an appealing mitigation, but we tested it across a *physical* device boundary and it did not transfer: forcing every scale to a power of two on our INT8 ResNet-50 left the x86↔A76 pair *further* from agreement than the unmodified baseline (958/1000 → 869/1000 for ceil rounding, 919/1000 for nearest), because these two MLAS kernels do not implement `M = 2^k` as a shared exact shift and the coarsened scale grid enlarges the epilogue divergence rather than removing it (§5). Whether it transfers across other boundaries (CPU↔accelerator, CPU↔vendor-NPU), or with a kernel pair that *does* share an exact power-of-two shift, remains open; on the evidence here, per-target re-validation — not a scale constraint — is the dependable path. These findings also motivate our follow-on work characterizing a heterogeneous multi-module automotive compute platform, where the inter-module data-movement bottleneck (a level up from §7) and cross-module INT8 consistency (a level up from §5) become first-order system design constraints.

**Artifact availability.** Measurement scripts and 32 HTML reports are released with this paper; Appendix A maps every numbered claim to the report and script that produced it. Available at <https://github.com/yyshin-katech/embedded-ai-quantization-guide/tree/paper1-v1>.

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

---

## Appendix A. Claim-to-Artifact Map

Every numbered claim in this paper is backed by a measurement report in `logs/` and by the scripts and result files that produced it in `experiments/`. Report names below are relative to `logs/`; artifact paths are relative to `experiments/`, both in the artifact repository linked above. Within an artifact cell, an entry that contains no slash is a file in the same directory as the first entry of that cell. Of the 32 reports released, the 24 cited here are the ones a claim in this paper depends on; the remaining eight cover the same corpus's supporting work (environment setup, PTQ deep-dive and raw run logs, QAT recovery, the BEVFormer/BEVDet capstone, and a CNN-detector accuracy axis) and are not load-bearing for any claim above. Claims are abbreviated here, and the absolute values they quote (latency, top-1, mAP, perf-per-watt) are batch-1, subset-based, and measured on different paths, so only the within-comparison *relative* relations they support are valid (§10).

| Sec. | Claim | Report (`logs/`) | Scripts and results (`experiments/`) |
|---|---|---|---|
| §4 (C1) | INT8 speedup sign set by the dot-product ISA, four CPUs, one model, one runtime | `stage4_arm_cpu_fallback_report.html`<br>`stage4_imx8mn_a53_report.html`<br>`stage4_jetson_agx_orin_a78ae_report.html` | `stage5_infrastructure/cpu_proxy/rpi_bench.py`<br>`rpi_bench_lowmem.py`<br>`stage5_infrastructure/cpu_proxy/results/resnet50__*.json` |
| §5 (C2) | FP32 1000/1000 control; CPU↔CPU INT8 agreement 1000 / 965 / 961 / 958 | `stage4_arm_cpu_fallback_report.html`<br>`stage4_imx8mn_a53_report.html`<br>`stage4_jetson_agx_orin_a78ae_report.html` | `stage5_infrastructure/cpu_proxy/README.md`<br>(the agreement matrix)<br>`stage5_infrastructure/cpu_proxy/raw/` |
| §5 (C2) | CPU↔accelerator 961/1000 from the same QDQ artifact; iGPU INT8 top-1 0.7620 | `stage3_jetson_orin_accuracy_report.html` | `stage3_tensorrt/jetson_ondevice/accuracy/scripts/orin_accuracy.py`<br>`analyze_accuracy.py` |
| §5 (C2 ‡) | Vendor-NPU 939/1000, scales not held fixed (deployment observation only) | `stage4_deepx_dxm1_accuracy_report.html` | `stage4_deepx_dxm1/accuracy/scripts/npu_infer.py`<br>`cpu_infer.py`<br>`analyze_dxm1_acc.py` |
| §5, §11 | Power-of-two mitigation does not transfer across the x86↔A76 boundary: 958 → 869 (ceil) / 919 (nearest), gate NO-GO; cross-kernel |Δlogit| ×4.34 / ×2.35, tie-breaking refuted; 30 runs bit-identical | `stage5_pot_scales_report.html` | `stage5_infrastructure/pot_scales/pot_rewrite.py`<br>`pot_bench.py`<br>`pot_agree.py`<br>`pot_mech.py`<br>`stage5_infrastructure/pot_scales/results/` |
| §6 (C3) | Qualcomm HTP silent BYO-QDQ failure 0.75 → 0.005; native path recovers 0.735 | `stage4_qualcomm_aihub_report.html` | `stage4_qualcomm_aihub/scripts/qaihub_int8.py`<br>`qaihub_native_quant.py`<br>`qaihub_acc.py` |
| §6 (C3) | DEEPX loud refusal (GraphStructureError, no engine); native path 0.7660 | `stage4_deepx_dxm1_accuracy_report.html` | `stage4_deepx_dxm1/accuracy/scripts/probe_extqdq.py`<br>`compile_dxm1.py` |
| §7 (C4) | Three models, two regimes on one device; YOLOv5s 26.3× slower than ResNet-50 | `stage4_deepx_dxm1_crossover_report.html` | `stage4_deepx_dxm1/crossover/scripts/analyze_profiler.py`<br>`build_crossover_summary.py` |
| §7 (C4) | Fixed-compute output sweep, 12 points; transition band width = core count | `stage4_deepx_dxm1_transition_report.html` | `stage4_deepx_dxm1/transition/scripts/build_transition_models.py`<br>`run_transition_bench.sh`<br>`build_transition_summary.py`<br>`stage4_deepx_dxm1/transition/results/transition_summary.json` |
| §7 (C4) | Third regime: DETR host-CPU-compute-bound stage decomposition | `stage4_deepx_dxm1_detr_report.html` | `stage4_deepx_dxm1/detr/scripts/npu_infer_detr.py`<br>`analyze_detr_regime.py`<br>`stage4_deepx_dxm1/detr/results/detr_dxm1_summary.json` |
| §7 | NPU vs. host CPU in the compute-bound regime: ×11.42 throughput, ×29.29 perf/W | `stage4_deepx_dxm1_report.html` | `stage4_deepx_dxm1/scripts/cpu_bench.py`<br>`analyze_profiler.py`<br>`build_summary.py` |
| §8.1 | DETR INT8 collapse −42.9%; op-selection no-op; two-way ablation | `stage2_detr_quantization_report.html` | `stage2_detr/s2_04_ptq.py`<br>`s2_07_coco_eval.py`<br>`s2_08_quantize_mixed.py`<br>`s2_09_quantize_ablation.py` |
| §8.1 | Jetson symmetric-requantization cross-confirmation −43.8% | `stage3_jetson_orin_detr_accuracy_report.html` | `stage3_tensorrt/jetson_ondevice/detr_accuracy/scripts/detr_sym_export.py`<br>`orin_detr_map.py` |
| §8.1 | SmoothQuant recovers 59.9% of the gap (torch fake-quant path) | `stage2_smoothquant_report.html` | `stage2_smoothquant/sq_01_modelopt_api.py`<br>`sq_03_absmax_smooth.py`<br>`sq_04_alpha_sweep.py` |
| §8.1 | SmoothQuant recovers only ~9% on-device (Gemm-only INT8 coverage) | `stage3_jetson_orin_detr_smoothquant_report.html` | `stage3_tensorrt/jetson_ondevice/detr_smoothquant/scripts/detr_sq_export.py`<br>`orin_detr_sq_map.py` |
| §8.2–§8.3 | DEEPX auto-split (no collapse), 23.99 MB handoff; two-quantizer comparison | `stage4_deepx_dxm1_detr_report.html` | `stage4_deepx_dxm1/detr/scripts/analyze_detr_map.py`<br>`build_detr_summary.py` |
| §8 (note) | NVDLA INT8-only datapath, 51.29 inf/s/W; DETR DLA fragmentation (16 ForeignNodes) | `stage3_jetson_orin_ondevice_report.html`<br>`stage3_jetson_orin_concurrent_power_report.html`<br>`stage3_jetson_orin_detr_report.html` | `stage3_tensorrt/jetson_ondevice/scripts/ppw.py`<br>`concurrent.py`<br>`power_sweep.py`<br>`stage3_tensorrt/jetson_ondevice/detr/scripts/detr_bench.py` |
| §9 (C8) | Subset inflation +9.77 pp (mean of 8 configs); preprocessing −1.07 pp | `stage1_50k_rerun_reproduction_report.html`<br>`stage1_real_imagenet_report.html` | (no separate script dir; procedure is in the report and in study_guide/03, /10) |
| §9 (C8) | Per-layer SQNR does not predict Δtop-1 (Spearman ρ = −0.030, 21 layers) | `stage1_50k_rerun_reproduction_report.html` | (no separate script dir; procedure is in the report and in study_guide/03) |
| §9 (C8) | TensorRT EP listed but silently on CPU, p50 11.83 → 0.41 ms once fixed | `stage0.5_ladder_log.html` | (no separate script dir; fix and probe are in study_guide/01) |
| §9 (C8) | Zero-copy output-buffer aliasing collapses top-1 to 0.0014, no error raised | `stage5_infrastructure_report.html` | `stage5_infrastructure/bench/run_bench.py`<br>`stage5_infrastructure/bench/report/generate.py`<br>`stage5_infrastructure/bench/tests/test_regression.py` |
| §9 (C8) | Opset down-convert “succeeds” (exit 0) while emitting an invalid graph | `stage1_quantization_log.html` | (no separate script dir; procedure is in the report) |
| §9 (C8) | pip TensorRT wheel ships without the ‘trtexec’ binary the tutorials assume | `stage3_tensorrt_report.html` | `stage3_tensorrt/t01_env.py`<br>`t02_latency_3point.py` |
