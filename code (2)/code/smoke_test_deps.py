import sys
import importlib

packages = [
    "sys",
    "platform",
    "python",
    "numpy",
    "torch",
    "torchvision",
    "torchaudio",
    "gym",
    "cv2",
    "matplotlib",
    "PIL",
    "scipy",
    "tqdm",
    "yaml",
    "requests",
]

checks = []

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

for pkg in packages:
    name = pkg
n
# explicit imports with nicer handling
to_import = {
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "gym": "gym",
    "cv2": "cv2",
    "matplotlib": "matplotlib",
    "PIL": "PIL",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "yaml": "yaml",
    "requests": "requests",
}

for key, module_name in to_import.items():
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", None)
        extra = ""
        if module_name == "torch":
            ver = getattr(mod, "__version__", "?")
            cuda = getattr(mod, "cuda", None)
            cuda_avail = cuda.is_available() if cuda is not None else "?"
            extra = f", cuda_available={cuda_avail}"
        if module_name == "PIL":
            ver = getattr(mod, "PILLOW_VERSION", None) or getattr(mod, "__version__", None)
        print(f"OK: {module_name} imported, version={ver}{extra}")
    except Exception as e:
        print(f"ERROR: Failed to import {module_name}: {e}")

# check torchvision/torchaudio can access torch
try:
    import torchvision
    print("torchvision ok")
except Exception as e:
    print("torchvision import error:", e)

try:
    import torchaudio
    print("torchaudio ok")
except Exception as e:
    print("torchaudio import error:", e)

print('\nSmoke test complete')
