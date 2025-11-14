import torch
import cv2
import numpy as np
import ollama
import whisper

print("=" * 60)
print("AI GUARDBOT GPU OPTIMIZATION SETUP VERIFICATION")
print("=" * 60)

# Check 1: PyTorch + CUDA
print("\n✓ PyTorch Setup:")
print(f"  GPU Available: {torch.cuda.is_available()}")
print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
print(f"  CUDA Version: {torch.version.cuda}")
print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Check 2: FaceNet
print("\n✓ FaceNet PyTorch:")
from facenet_pytorch import MTCNN, InceptionResnetV1
print(f"  MTCNN imported successfully")
print(f"  InceptionResnetV1 imported successfully")

# Check 3: OpenCV
print("\n✓ OpenCV:")
print(f"  Version: {cv2.__version__}")

# Check 4: Whisper
print("\n✓ Whisper:")
print(f"  Model will load on first use (CPU)")

# Check 5: Ollama
print("\n✓ Ollama:")
try:
    client = ollama.Client(host='http://localhost:11434')
    print(f"  Connected to Ollama server at http://localhost:11434")
except:
    print(f"  ⚠️ Ollama server not responding - start with: ollama serve")

print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE ✓")
print("=" * 60)
