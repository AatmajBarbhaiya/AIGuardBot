🤖 AI Guard Bot - Jarvis Security System
An intelligent, multi-modal AI security system that combines voice activation, real-time face recognition, and conversational AI to monitor and respond to intruders. Named "Jarvis," the system uses GPU-optimized deep learning models for efficient real-time operation.

🎯 Features
Voice Activation (Milestone 1)
Whisper-based speech recognition with anti-hallucination safeguards

Fuzzy command matching using sequence similarity (70%) and word intersection (30%)

Activation phrases: "guard mode", "activate guard", "start security"

False activation prevention with cooldown timers

Face Recognition (Milestone 2)
Real-time face detection using FaceNet InceptionResnetV1

18-20 FPS performance on RTX 4050 GPU

Cosine similarity matching with 0.75 confidence threshold for trusted faces

Multi-face simultaneous detection and classification

Thread-safe camera and face processing

LLM Escalation Protocol (Milestone 3)
3-level escalation system for unknown intruders:

Level 1 (2s): Initial contact + identity verification

Level 2 (4s): Context-aware warning + exit demand

Level 3 (6s): Final warning + authority notification

Jarvis personality system (Stern Guard, Professional Security, Friendly Guard)

Context tracking with name/purpose extraction

20-word response limit for concise communication

Anti-hallucination prompt engineering

System Features
100% GPU utilization (6.4GB VRAM optimized)

State machine architecture with clean transitions

Graceful shutdown with Ctrl+C handling

Thread-safe operations across audio, video, and face processing

Accumulated timer with pause/reset logic for presence tracking

🖥️ Hardware Requirements
Minimum Requirements
GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3050 or better)

CPU: Quad-core processor (Intel i5/AMD Ryzen 5 or better)

RAM: 8GB system memory

Storage: 10GB free space for models

Camera: USB webcam or built-in camera

Microphone: Any USB/built-in microphone

Recommended Configuration
GPU: NVIDIA RTX 4050/4060 (6-8GB VRAM)

CPU: Hexa-core or better

RAM: 16GB

Camera: 720p or 1080p webcam

Microphone: Noise-canceling USB microphone

Tested Hardware
NVIDIA GeForce RTX 4050 Laptop GPU (6.4GB VRAM)

Intel Core i7-12700H

16GB DDR4 RAM

Windows 11

📦 Installation
Step 1: Clone the Repository
bash
git clone https://github.com/AatmajBarbhaiya/ai-guard-bot.git
cd ai-guard-bot

text

### Step 2: Install CUDA Toolkit

Download and install **CUDA 12.1** from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

Verify installation:
``````bash
nvcc --version
nvidia-smi
Step 3: Install Ollama
Download and install Ollama from ollama.ai.

Pull the Llama3.1 model:
bash
ollama pull llama3.1:8b

text

Verify:
``````bash
ollama list
Step 4: Install Python Dependencies
bash
pip install -r requirements.txt

text

**Key dependencies:**
- torch==2.0.1+cu121 (PyTorch with CUDA 12.1)
- torchaudio==2.0.2+cu121
- torchvision==0.15.2+cu121
- openai-whisper==20231117
- facenet-pytorch==2.5.3
- opencv-python==4.8.1.78
- pyttsx3==2.90
- pyaudio==0.2.14
- ollama==0.1.6
- numpy<2.0

### Step 5: Install PyTorch with CUDA Support

``````bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Verify GPU access:
python
import torch
print(torch.cuda.is_available()) # Should print: True
print(torch.version.cuda) # Should print: 12.1

text

---

## 🚀 Quick Start

### 1. Enroll Trusted Faces

Before running the guard bot, enroll trusted users:

python face_enrollment.py
Follow the prompts:

Enter your name (e.g., "aatmaj")

Position your face in the camera frame

Press 's' to capture (capture 3-5 images from different angles)

Press 'q' to finish

Faces are saved to data/embeddings/known_faces.pkl.

2. Run the Guard Bot
python main.py

text

### 3. Select Personality

Choose from:
1. **Stern Guard** - Direct, no-nonsense
2. **Professional Security** - Formal, protocol-focused
3. **Friendly Guard** - Polite but firm

### 4. Activate Guard Mode

Say one of the activation phrases:
- "guard mode"
- "activate guard"
- "start security"

The system will confirm: *"Guard mode activated"*

### 5. Deactivate

With a **trusted face visible**, say:
- "deactivate guard"
- "stop guard"

The system requires BOTH voice command AND trusted face for deactivation.

---

## 📁 Project Structure

AI-Agent-Bot/
├── data/
│ ├── embeddings/
│ │ └── known_faces.pkl # Enrolled face embeddings
│ ├── test_images/ # Test images for validation
│ └── trusted_faces/ # Trusted face image storage
├── logs/
│ └── agent.log # System logs
├── milestones/
│ ├── milestone1_voice_activation.py # M1: Voice activation
│ ├── milestone2_face_recognition.py # M2: Face recognition
│ └── milestone3_llm.py # M3: LLM integration
├── utils/
│ ├── audio_utils.py # Audio processing & VAD
│ ├── config.py # Configuration constants
│ ├── conversation_manager.py # Context tracking
│ ├── numpy_compat.py # NumPy compatibility fixes
│ ├── state_manager.py # State machine logic
│ └── video_utils.py # Video capture utilities
├── main.py # Main application entry point
├── face_enrollment.py # Face enrollment script
├── requirements.txt # Python dependencies
└── README.md # This file

text

***

## ⚙️ Configuration

### Escalation Timing (Edit `main.py`)

```python```
ESCALATION_TIMING = {
    'level1_presence': 2,   # Trigger L1 after 2s
    'level2_presence': 4,   # Trigger L2 after 4s
    'level3_presence': 6,   # Trigger L3 after 6s
    'face_loss_reset': 7,   # Reset if absent >7s
    'cooldown_duration': 30 # Cooldown after L3
}
Face Recognition Thresholds
python
FACE_DISTANCE_THRESHOLD = 0.40  # Cosine distance threshold
CONFIDENCE_THRESHOLD = 0.75     # Minimum confidence for "trusted"
```

### Whisper Anti-Hallucination Settings

```python```
whisper_model.transcribe(
    audio,
    no_speech_threshold=0.8,          # Silence detection
    logprob_threshold=-0.5,           # 60% confidence required
    condition_on_previous_text=False, # No context bias
    temperature=0.0                   # Deterministic output
)
```

---

## 🔍 How It Works

### State Machine Flow

IDLE → M1 (Voice) → M2 (Face) → M3 (Escalation) → IDLE

text

1. **IDLE:** Personality selection, model loading
2. **M1_VOICE:** Continuous audio monitoring for activation phrase
3. **M2_FACE:** Real-time face recognition + deactivation monitoring
4. **M3_ESCALATION:** 3-level protocol for unknown intruders

### GPU Memory Allocation

**Loading Sequence:** Ollama → FaceNet → Whisper

|| Model | VRAM | Percentage |
|-------|------|------------|
| Ollama (Llama3.1:8b) | 4.7 GB | 73.4% |
| Whisper Medium | 1.5 GB | 23.4% |
| FaceNet | 0.2 GB | 3.1% |
| **Total** | **6.4 GB** | **100%** |

This specific order prevents CUDA out-of-memory errors by loading Ollama before PyTorch claims GPU memory.

***

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Ensure models load in correct order (Ollama first). Restart Python kernel and try again.

Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

text

### Issue: Whisper hallucinations (e.g., "Thanks for watching!")

**Solution:** Already fixed with stricter parameters. If persists, increase `no_speech_threshold` to 0.9.

### Issue: TTS echo causing false deactivation

**Solution:** Audio thread automatically pauses during TTS. If issue persists, increase pause duration in `main.py`:

``````python
self.audio_thread.stop_listening()
time.sleep(0.5)  # Increase from 0.3 to 0.5
Issue: Face recognition too slow (<15 FPS)
Solution:

Ensure GPU is being used: self.device = torch.device("cuda:0")

Enable cuDNN optimization: torch.backends.cudnn.benchmark = True

Process every 2nd frame instead of every frame

Issue: LLM hallucinating names
Solution: Already fixed with explicit anti-hallucination prompts. Check that context.person_name is properly extracted from speech.

Issue: PyTorch not detecting GPU
Solution: Reinstall PyTorch with correct CUDA version:

bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

text

---

## 📊 Performance Metrics

- **Face Recognition FPS:** 18-20 FPS
- **Voice Activation Latency:** <0.5s
- **Whisper Transcription:** ~0.5s per 3s audio chunk
- **LLM Response Time:** 2-4s
- **Face Recognition Accuracy:** 89% confidence on known faces
- **Voice Command Similarity:** 0.84-0.89 on valid commands

---

## 🔒 Security Features

- **Recording notice** in Level 1 responses (legal compliance)
- **Dual-factor deactivation** (voice + trusted face)
- **30-second cooldown** after Level 3 prevents spam
- **Accumulated timer** prevents false escalations from brief appearances
- **Context tracking** maintains conversation history across levels

---

## 🎓 Academic Context

This project was developed as part of **EE782 - Advanced Machine Learning** coursework, focusing on:
- Multi-modal AI integration
- GPU memory optimization
- Real-time inference optimization
- State machine design patterns
- Prompt engineering for LLMs

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **FaceNet (PyTorch implementation)** for face recognition
- **Ollama & Meta's Llama 3.1** for conversational AI
- **NVIDIA CUDA** for GPU acceleration
- Course instructor and TAs for guidance (EE782- Advance Machine Learning)

---

## 📞 Contact

**Developer:** Aatmaj  
**Contact:** aatmaj017@gamil.com  
**Year:** 2025

---

**Built with ❤️ using PyTorch, Whisper, FaceNet, and Llama 3.1**