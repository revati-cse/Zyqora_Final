# Model comparison

All figures below were **measured on the x86_64 development host** (shared cloud VM, 4 vCPU Intel Xeon @ 2.1 GHz, Python 3.12, ExecuTorch 1.3.1).
**None of them are Raspberry Pi 5 results.** Host timings were noisy: the same configuration varied by up to about 2× between runs.
Replace this table with `scripts/benchmark_pi.py` output from the Pi before quoting any latency.

## Candidates

| | MobileNetV2 FP32 | MobileNetV2 INT8 (head FP32) | MobileNetV2 INT8 (full) | ShuffleNetV2 x1.0 INT8 (Arm Portal) | CLIP ViT-B/32 INT8 |
|---|---|---|---|---|---|
| Status | **Built, verified, default** | Built, verified | Built, verified | Not tested | Not implemented (contingency) |
| Artifact | `mobilenetv2_fp32_xnnpack.pte` | `mobilenetv2_int8_xnnpack.pte` | `mobilenetv2_int8full_xnnpack.pte` | portal download | portal download |
| Size (measured) | 13.99 MB | 7.59 MB | 3.76 MB | ~2.4 MB *(published)* | ~327.4 MB *(published, per brief)* |
| XNNPACK delegation | 1 delegate, entire graph (only `getitem` outside) | same | same | – | – |
| Top-1 vs PyTorch, 100 course images | **100/100**, max abs logit diff 4.4e-5 | – | – | – | – |
| Top-1 vs FP32, 20 held-out images | – | **16/20 (80%)**, max abs logit diff 4.48 | 15/20 (75%), max abs logit diff 4.51 | – | – |
| Host model-only median, 1 thread | 9.5 ms (5.9 ms in an earlier run) | 4.9 ms | 4.5–9.3 ms (noisy) | 3.63 ms p50 *(published, Pi 5)* | not measured |
| Host model-only median, 4 threads | 2.7–3.5 ms | 1.9–2.1 ms | 1.9–2.1 ms | – | – |
| Host CPU time / inference, 1 thread | 9.4 ms | 5.2 ms | 4.6 ms | – | – |
| Host CPU time / inference, 4 threads | 10–12.6 ms | 7.1–7.6 ms | 7.1–7.8 ms | – | – |
| Peak RSS increase over runtime import* | 37.6 MB | 27.1 MB | – | 9.22 MB peak *(published)* | – |
| ImageNet top-1 | 71.88% *(TorchVision published, FP32)* | not measured | not measured | 68.76% *(published)* | – |
| Organiser-category accuracy | **pending organiser data** | pending | pending | – | – |

\* `VmHWM` at the end of the benchmark minus `VmRSS` after importing torch and ExecuTorch. It includes model load, XNNPACK weight packing, inference buffers, and the end-to-end loop's JPEG decoding. Importing torch and ExecuTorch alone added about 642 MB RSS on this host (CUDA-enabled PyPI torch wheel); the Pi's CPU-only aarch64 wheel will differ and must be measured.

## Notes per model

**MobileNetV2 FP32 (baseline, default).** These are the official TorchVision weights, exported with the exact Arm course Module 2 sequence. It passes the FP32 parity gate: identical top-1 on all 100 Open Images course calibration images. Zyqora's NumPy preprocessing is bit-identical to the TorchVision transform (max diff 0.0), so the Pi does not need TorchVision.

**MobileNetV2 INT8.** PT2E static quantization with `XNNPACKQuantizer` (symmetric, per-channel), following the course. It was calibrated on 80 course calibration images, and the 20 held-out images were verified to be disjoint by SHA-256. The course default keeps `classifier.1` (1280×1000 weights, about 5 MB) in FP32, which is why it is 7.6 MB rather than about 3.5 MB. The `--quantize-classifier-head` variant is 3.76 MB. Both changed the top-1 class on 4–5 of 20 held-out images, including confident ones (e.g. "motor scooter" became "triumphal arch"). INT8 was faster and used less CPU per inference on the host, but at this agreement level it **is not selected**. Re-evaluate it on the Pi with organiser-category accuracy. If the INT8 *category* accuracy equals FP32 on the organiser/dev set, its lower CPU time per image makes it the better choice.

**ShuffleNetV2 x1.0 INT8 (backup).** These are published Arm Portal figures only (built with ExecuTorch 1.1.0, while this project pins 1.3.1). It was not downloaded or tested, and cross-version `.pte` compatibility is not assumed. Use it only if MobileNetV2 cannot meet Pi constraints. It would need a separate check of its preprocessing and label order, and possibly a separate runtime environment.

**CLIP ViT-B/32 INT8 (contingency).** Not implemented, per the brief: it is about 23× larger than the FP32 MobileNetV2 and would need an offline tokenizer and text embeddings, and its Pi latency and waste accuracy are unknown. Consider it only if the organiser categories cannot be represented by ImageNet classes (e.g. "e-waste" vs "hazardous" distinctions that ImageNet lacks). In that case, verify the artifact, licence, runtime version, tokenizer, offline execution, latency and memory first.

**Object detection (YOLO).** Not implemented. There is no evidence yet that the organiser images contain multiple objects.

## Selection rule and current decision

Choose the artifact by **organiser-category accuracy first**, then Pi latency and CPU time per image, memory, size, and reliability. Do not choose by published ImageNet accuracy.

**Current decision: MobileNetV2 FP32** (`config/app.yaml → model.pte`). It is the only artifact with a passed parity gate, and the INT8 agreement with FP32 is too low to adopt without organiser-category evidence. Revisit after:
1. `run_zyqora.py` on the organiser/dev set with both artifacts (compare `overall_accuracy`).
2. `scripts/benchmark_pi.py --threads 1 2 4` for both artifacts on the Pi.

### Thread count
On the host, more threads lowered latency but raised CPU time per inference. For FP32: 1 thread ≈ 9.4 ms CPU/inference, 4 threads ≈ 10–12.6 ms CPU/inference, at about 3× lower latency. For a one-image-at-a-time sorting station either is fast enough. Pick the Pi setting from measured data: lowest CPU time per image if latency is already well under human reaction time, otherwise lowest median latency.
