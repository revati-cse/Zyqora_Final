import platform, sys
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Machine:", platform.machine())
try:
    import torch, torchvision
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
except Exception as exc:
    print("Torch/torchvision unavailable:", repr(exc))
try:
    import executorch
    print("ExecuTorch:", getattr(executorch, "__version__", "imported"))
except Exception as exc:
    print("ExecuTorch unavailable:", repr(exc))
