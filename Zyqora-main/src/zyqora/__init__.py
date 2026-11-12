"""Zyqora AI: offline, confidence-aware waste sorting on Raspberry Pi 5.

Components:
    preprocessing  - deterministic MobileNetV2 input contract (Pillow + NumPy)
    model_runtime  - ExecuTorch .pte loader / runner (XNNPACK)
    materialwise   - ImageNet prediction -> organiser category -> disposal action
    ecoguard       - ACCEPT / REVIEW decision layer
    voicesort      - offline Tamil / English instructions (pre-generated clips)
    greenscore     - supervisor-verified sorting rewards (SQLite)
    evaluator      - mandatory accuracy table and summary
    benchmark      - latency / CPU time / memory measurement helpers
"""

__version__ = "1.0.0"
