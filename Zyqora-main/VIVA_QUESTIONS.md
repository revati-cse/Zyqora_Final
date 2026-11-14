# Project viva: likely questions and honest answers

**1. Why use YOLO and MobileNetV2 together instead of one model?**
YOLO localises several objects, but the shipped weights only know 80 COCO classes, which are not waste categories. MobileNetV2 was fine-tuned on the organiser's Organic / Recyclable images, but labels one image at a time. Detect → crop → classify gives one waste label per object (ARCHITECTURE.md).

**2. Is your YOLO trained on waste?**
No. It is the pretrained COCO YOLO11n. We use it as an object-proposal stage, and its labels only as a conservative material hint. The waste-detector training pipeline (TACO) is implemented, but it was not run: no access to the images, no GPU, and retraining needs team approval (DETECTION.md).

**3. Why YOLO11n at 640 px?**
It was measured on 4,952 held-out VOC2007 test images:
- mAP@0.5 is 0.832 for YOLO11n vs 0.813 for YOLOv8n;
- at 320 px it drops to 0.762, and bottle AP50 falls from 0.772 to 0.622;
- YOLO11n was also the faster of the two on our host.

Pi 5 latency still has to be measured.

**4. What is your detection accuracy on waste?**
Not measured. There was no box-annotated waste test set; the VOC numbers are for generic objects. We do not claim a waste mAP.

**5. Classifier accuracy?**
96.65% on the team's 2,000-image validation set. This was reproduced on the Pi 5 as 96.60% (1,932 / 2,000). With EcoGuard at 0.95, 10.8% of items go to review and accepted-only accuracy is 98.71%. This is validation, not an independent TEST set; TEST is pending.

**6. What does the model do with a battery or a TV?**
The classifier has no Residual or e-waste class, so it forces such an item into Organic or Recyclable, sometimes at 100% confidence. When the detector recognises an electronic device (tv, laptop, phone, …), the rules route it to e-waste collection and show a warning. Electronics the detector does not recognise are the known gap.

**7. How is the Green Score computed? Is it scientific?**
RP × 0.40 + EI × 0.35 + DS × 0.25, using 0–100 ratings per material that the team chose, each with a written reason. It is a transparent, configurable indicator, not an LCA and not a measurement. It never uses model confidence, and unreliable items are not scored (GREEN_SCORE.md).

**8. Why do some items show "Unknown"?**
Either EcoGuard was not confident enough (score below 0.95), or the detector and classifier disagreed (e.g. "banana" classified Recyclable). We prefer "ask a supervisor" to a confident wrong bin.

**9. How do organiser-defined categories work?**
The model outputs Organic or Recyclable: the organiser dataset's own labels. Profiles rename them one-to-one:
- Recyclable / Non-Recyclable;
- Food / Waste.

The model label is always stored separately, and a profile cannot invent a class the model does not have.

**10. What runs on the Raspberry Pi and how fast?**
Both models run on the CPU through ExecuTorch with XNNPACK.
- **Classifier:** 12.7 ms median with 2 threads, measured on the Pi.
- **Detector and full pipeline:** host figures exist (BENCHMARK_REPORT.md), but the Pi numbers must come from `scripts/benchmark_detector.py` on the Pi. We do not extrapolate.

**11. Do you measure power?**
No. CPU load is not converted to watts. On a Pi 5 the dashboard shows the PMIC's own rail readings (`vcgencmd pmic_read_adc`), labelled as rails-only. Real power needs an inline USB-C meter.

**12. What is the latency cost of the two-stage design?**
One detector pass, plus one classifier pass per waste crop (the `.pte` has a fixed batch of 1). It is capped by `max_objects_classified`. On the host, 2 threads: about 64 ms YOLO + 9 ms per crop.

**13. Does it work offline?**
Yes. All models and rules are local files and nothing is downloaded at runtime; the dashboard is served by the device itself.

**14. How do you prevent data leakage in evaluation?**
- The detector was chosen on VOC2007 test, which COCO models never trained on.
- TACO splits are made by image, with a fixed seed.
- `scripts/check_leakage.py` compares classifier splits by exact file hash (SHA-256) and perceptual hash (dHash).
- The EcoGuard threshold came from validation, never from TEST.

**15. What are the main limitations?**
- The detector is not trained on waste, so cans, bags and cardboard are often missed.
- The classifier has only two classes.
- There is no end-to-end mixed-waste accuracy yet.
- There are no Pi detector benchmarks yet.
- Green Score ratings are assumptions.
- Windows support is unverified.
- Physical camera, speaker and actuator tests are pending.
