import torch
import subprocess
import sys

print("="*60)
print("🔍 PYTORCH + CUDA ENVIRONMENT DIAGNOSTICS")
print("="*60)

# ---------------------------
# Check PyTorch Installation
# ---------------------------

print("\n📌 Checking PyTorch installation...")

try:
    print(f"PyTorch Version: {torch.__version__}")
except Exception as e:
    print("❌ PyTorch not installed!")
    sys.exit(0)

# ---------------------------
# CUDA Availability
# ---------------------------

print("\n📌 Checking CUDA availability in PyTorch...")

if torch.cuda.is_available():
    print("✅ CUDA is available in PyTorch!")
    print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
else:
    print("❌ CUDA is NOT available in PyTorch!")

# ---------------------------
# GPU Info
# ---------------------------

print("\n📌 Checking GPU details...")

if torch.cuda.is_available():
    try:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs Detected: {device_count}")

        for i in range(device_count):
            print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")

        print(f"Current Device Index: {torch.cuda.current_device()}")
    except Exception as e:
        print("⚠️ Error while fetching GPU info:", str(e))
else:
    print("No GPU available to query.")

# ---------------------------
# CUDA Toolkit (nvcc)
# ---------------------------

print("\n📌 Checking CUDA Toolkit (nvcc)...")

try:
    output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    print("CUDA Toolkit (nvcc) is installed:")
    print(output)
except Exception:
    print("❌ nvcc (CUDA toolkit) NOT found in PATH (this is fine unless you're compiling CUDA kernels).")

# ---------------------------
# cuDNN Availability
# ---------------------------

print("\n📌 Checking cuDNN...")

try:
    print("cuDNN Enabled:", torch.backends.cudnn.enabled)
    print("cuDNN Version:", torch.backends.cudnn.version())
except Exception as e:
    print("❌ cuDNN not available or not detected!", str(e))

print("\n" + "="*60)
print("🎉 DIAGNOSTICS COMPLETE")
print("="*60)
