"""Report the local runtime without installing, downloading or starting jobs."""
import json
import platform
from importlib.metadata import version, PackageNotFoundError
import torch
names = ("numpy", "scipy", "torch", "PyYAML", "pytest", "fastapi", "transformers", "peft", "accelerate", "bitsandbytes")
packages = {}
for name in names:
    try: packages[name] = version(name)
    except PackageNotFoundError: packages[name] = None
print(json.dumps({"python": platform.python_version(), "packages": packages,
      "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "qwen_weights_loaded": False}, indent=2))
