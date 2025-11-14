"""
Milestone 3: GPU-ACCELERATED LLM GUARD SYSTEM
- Whisper on CPU (GPU hidden) - 700MB
- Ollama llama3.1:8b on GPU - 4700MB
- Dynamic context-aware responses with personality system
- Real-time CPU/GPU monitoring (pynvml for accurate GPU)
- Listening: 5s non-VAD timer (auto-skip) + VAD switch for full speech (no cutoffs)
- TTS: Print before speaking (flush ensured)
"""

import os

# === FORCE WHISPER TO CPU ONLY - MUST BE FIRST ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from Whisper completely

import time
import logging
import random
import re
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import psutil  # For CPU monitoring
import io
import torch
import ollama
from collections import deque  # ✅ ADD THIS LINE

# pynvml for accurate GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml not available - Install with 'pip install nvidia-ml-py' for GPU tracking")

# Direct pyaudio for non-VAD timer
try:
    import pyaudio
    PY_AUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PY_AUDIO_AVAILABLE = False
    print("⚠️ pyaudio not available - auto-skip always")

# === UTF-8 ENCODING FIX FOR WINDOWS ===
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, TypeError):
        pass

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "utils"))

try:
    from utils.conversation_manager import ConversationManager
    from utils.audio_utils import StreamVAD, VADParams, int16_to_float32
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback imports if utils missing
    ConversationManager = None
    class StreamVAD:
        def __init__(self, params): pass
        def stream_chunks(self): yield None
        def stop(self): pass
    VADParams = lambda **kwargs: None
    def int16_to_float32(data): return data.astype(np.float32) / 32768.0

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configurable escalation timing
ESCALATION_TIMING = {
    'level1_presence': 2,
    'level2_presence': 4,
    'level3_presence': 6,
    'face_loss_reset': 10
}

# Hardcoded trusted person greetings
TRUSTED_GREETINGS = {
    "morning": [
        "Good morning! Welcome back.",
        "Morning! Everything's secure.",
        "Hello! Good to see you this morning."
    ],
    "afternoon": [
        "Good afternoon! Welcome back.",
        "Afternoon! All clear here.",
        "Hello! Hope you're having a good day."
    ],
    "evening": [
        "Good evening! Welcome back.",
        "Evening! Room is secure.",
        "Hello! Everything's in order this evening."
    ],
    "generic": [
        "Welcome back!",
        "Hello! Good to see you.",
        "Access granted. Welcome back."
    ]
}

# === CPU/GPU MONITORING CLASS ===
class DeviceMonitor:
    """Monitor CPU and GPU usage in real-time"""

    @staticmethod
    def get_cpu_usage():
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)

    @staticmethod
    def get_gpu_usage():
        """Get GPU usage via pynvml (accurate for Ollama/PyTorch)"""
        if not PYNVML_AVAILABLE:
            return {'available': False}

        try:
            pynvml.nvmlInit()  # Initialize NVML (safe to call multiple times)
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                pynvml.nvmlShutdown()
                return {'available': False}

            # Use first GPU (index 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = memory_info.total / (1024**3)  # Convert bytes to GB
            used_gb = memory_info.used / (1024**3)

            pynvml.nvmlShutdown()  # Cleanup
            return {
                'available': True,
                'memory_used_gb': used_gb,
                'memory_total_gb': total_gb
            }
        except Exception as e:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            return {'available': False, 'error': str(e)}

    @staticmethod
    def print_device_status(operation=""):
        """Print current device usage"""
        cpu = DeviceMonitor.get_cpu_usage()
        gpu = DeviceMonitor.get_gpu_usage()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n📊 [{timestamp}] [{operation}]")
        print(f"   CPU: {cpu:.1f}%")
        
        if gpu['available']:
            print(f"   GPU: {gpu['memory_used_gb']:.2f}GB / {gpu['memory_total_gb']:.1f}GB")
        else:
            print("   GPU: Unavailable")

class RobustTTSEngine:
    """Robust TTS with per-call engine reinitialization (Windows-safe)"""
    
    def __init__(self):
        """Minimal init - engine created fresh per call"""
        pass
    
    def speak(self, text: str):
        """Speak text with fresh engine each time (Windows audio lock workaround)"""
        if not text or not text.strip():
            return
        
        # Always print first (immediate feedback)
        print(f"🗣️ GUARD: {text}", flush=True)
        
        # ✅ KEY FIX: Create FRESH engine for EACH call
        try:
            import pyttsx3
            import time
            
            # Create brand new engine
            engine = pyttsx3.init()
            
            # Configure
            engine.setProperty('rate', 165)  # Slightly faster
            engine.setProperty('volume', 0.95)
            
            # Speak
            engine.say(text)
            engine.runAndWait()
            
            # ✅ CRITICAL: Force cleanup
            engine.stop()
            del engine
            
            # ✅ CRITICAL: Grace period for Windows audio device release
            time.sleep(0.5)
            
        except Exception as e:
            logger.warning(f"⚠️ TTS error: {e}")
            # Fallback: Silent pause (user reads printed text)
            import time
            time.sleep(max(2, len(text) * 0.04))

class SmartWhisperListener:
    """CPU-only Whisper listener with non-VAD 5s timer + VAD switch"""

    def __init__(self, whisper_device="cpu"):
        self.whisper_device = whisper_device
        self.whisper_model = None
        self.current_vad = None
        self._load_whisper()
        logger.info(f"🎤 SmartWhisperListener on {self.whisper_device}")

    def _load_whisper(self):
        """Load Whisper model on CPU only"""
        try:
            import whisper
            logger.info(f"📥 Loading Whisper base on CPU (GPU hidden)...")
            DeviceMonitor.print_device_status("BEFORE Whisper Load")
            
            self.whisper_model = whisper.load_model("base", device=self.whisper_device)
            
            DeviceMonitor.print_device_status("AFTER Whisper Load")
            logger.info(f"✅ Whisper loaded on CPU (700MB)")
        except Exception as e:
            logger.error(f"❌ Whisper loading failed: {e}")
            raise

    def _create_vad(self):
        """Create VAD instance"""
        try:
            vad_params = VADParams(
                frame_ms=30,
                pre_roll_ms=200,
                post_roll_ms=300,
                silence_threshold=0.003,
                min_speech_ms=800
            )
            return StreamVAD(vad_params)
        except Exception as e:
            logger.error(f"❌ VAD creation error: {e}")
            return None

    def transcribe_audio(self, audio_chunk):
        """Transcribe using CPU Whisper"""
        if not self.whisper_model or audio_chunk is None:
            return ""

        try:
            print("\n[TRANSCRIPTION START]")
            DeviceMonitor.print_device_status("BEFORE Transcription")
            
            audio_float = int16_to_float32(audio_chunk)
            result = self.whisper_model.transcribe(
                audio_float,
                fp16=False,  # CPU only - CRITICAL
                language='en',
                no_speech_threshold=0.7,
                logprob_threshold=-0.5
            )
            
            DeviceMonitor.print_device_status("AFTER Transcription")
            print("[TRANSCRIPTION END]\n")
            
            text = result.get("text", "").strip()
            if len(text) < 3:
                return ""
            return text
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""

    def listen_with_simple_timing(self, level: int, previous_speech: str = ""):
        """
        IMPROVED: 2-Phase listening with proper silence detection (2s timeout)
        
        - Phase 1: 5s countdown with continuous audio buffering (captures words BEFORE detection)
        - Phase 2: Direct audio collection until 2s CONSECUTIVE silence detected
        """
        
        if level == 3:
            print("🚨 LEVEL 3: No listening (final warning issued)")
            return "", datetime.now().strftime("%H:%M:%S")

        COUNTDOWN_TIME = 5
        ENERGY_THRESHOLD = 500
        SILENCE_TIMEOUT = 2.0  # 2 seconds of silence to stop listening
        MAX_LISTEN_DURATION = 10.0  # Max 10 seconds total (safety limit)
        
        print(f"\n🎤 LEVEL {level} LISTENING")
        if previous_speech:
            print(f"💬 Previous: '{previous_speech}'")
        print(f"💡 Speak now ({COUNTDOWN_TIME}s to respond or auto-skip)...")

        if not PY_AUDIO_AVAILABLE:
            print(f"⏰ [AUTO-SKIP] No audio input - escalating")
            return "", datetime.now().strftime("%H:%M:%S")

        audio_buffer = np.array([], dtype=np.int16)
        pa = None
        stream = None
        speech_detected = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        transcription = ""

        try:
            pa = pyaudio.PyAudio()
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000

            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            # ===== PHASE 1: 5s COUNTDOWN (CONTINUOUS BUFFERING) =====
            print("[DEBUG] PHASE 1: 5s countdown with continuous audio capture")
            
            for second in range(COUNTDOWN_TIME, 0, -1):
                print(f"⏰ {second}s...", end='', flush=True)
                second_start = time.time()
                
                while (time.time() - second_start) < 1.0:
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        chunk = np.frombuffer(data, dtype=np.int16)
                        audio_buffer = np.append(audio_buffer, chunk)  # ✅ KEEP ALL AUDIO
                        
                        if not speech_detected and np.max(np.abs(chunk)) > ENERGY_THRESHOLD:
                            speech_detected = True
                            print(" 🔊 Speech detected!", flush=True)
                    
                    except Exception as e:
                        logger.warning(f"Phase1 read error: {e}")
                        time.sleep(0.01)

                remaining = 1.0 - (time.time() - second_start)
                if remaining > 0:
                    time.sleep(remaining)

            print("\n[DEBUG] PHASE 1 done")

            if not speech_detected:
                print(f"⏰ [AUTO-SKIP] No speech in {COUNTDOWN_TIME}s - escalating")
                return "", timestamp

            print(f"✅ Speech detected! Phase 1 buffer: {len(audio_buffer)} samples ({len(audio_buffer)/RATE:.2f}s)")

            # ===== PHASE 2: DIRECT COLLECTION UNTIL 2s SILENCE =====
            print("\n[DEBUG] PHASE 2: Direct audio collection until 2s silence detected")
            print("⏰ Listening for completion (max 10s)...", flush=True)

            phase2_audio = []
            last_speech_time = time.time()
            phase2_start = time.time()
            frame_count = 0
            silence_buffer = deque(maxlen=int(SILENCE_TIMEOUT * RATE / CHUNK))  # Track silence duration

            while (time.time() - phase2_start) < MAX_LISTEN_DURATION:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    chunk = np.frombuffer(data, dtype=np.int16)
                    phase2_audio.append(chunk)
                    frame_count += 1

                    # Energy analysis
                    energy = np.max(np.abs(chunk))
                    has_speech = energy > ENERGY_THRESHOLD

                    if has_speech:
                        last_speech_time = time.time()
                        silence_buffer.clear()
                    else:
                        # Track silence duration
                        silence_buffer.append(1)

                    # Check if we have 2s of consecutive silence
                    silence_duration = time.time() - last_speech_time

                    if silence_duration >= SILENCE_TIMEOUT and frame_count > 20:  # At least 20 frames of speech
                        print(f"\n[DEBUG] Silence detected: {silence_duration:.2f}s >= {SILENCE_TIMEOUT}s")
                        print("✅ Speech complete (2s silence threshold reached)")
                        break

                    # Debug output every 50 frames
                    if frame_count % 50 == 0:
                        elapsed = time.time() - phase2_start
                        print(f"[DEBUG] Phase2: {frame_count} frames, {elapsed:.1f}s, silence: {silence_duration:.1f}s", flush=True)

                except Exception as e:
                    logger.warning(f"Phase2 read error: {e}")
                    break

            print(f"[DEBUG] PHASE 2 done: {frame_count} frames collected")

            # ===== PHASE 3: COMBINE & TRANSCRIBE =====
            print("\n[DEBUG] PHASE 3: Combining audio and transcribing")

            if phase2_audio:
                phase2_combined = np.concatenate(phase2_audio)
                combined_audio = np.append(audio_buffer, phase2_combined)
                print(f"[DEBUG] Total audio: {len(combined_audio)} samples ({len(combined_audio)/RATE:.2f}s)")
            else:
                combined_audio = audio_buffer
                print(f"[DEBUG] No Phase2 audio, using Phase1 only: {len(combined_audio)} samples")

            # Transcribe
            transcription = self.transcribe_audio(combined_audio)

            if transcription:
                print(f"🎤 INTRUDER: '{transcription}'")
                logger.info(f"🎤 Intruder speech: '{transcription}'")
            else:
                print("❓ No clear speech detected")

        except Exception as e:
            logger.error(f"❌ Listening error: {e}")
            print(f"❌ Listening error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            print(f"[DEBUG] Cleanup")
            if stream:
                stream.stop_stream()
                stream.close()
            if pa:
                pa.terminate()
            time.sleep(0.2)

        return transcription, timestamp


# [Rest of classes unchanged: FacePresenceTracker, ConversationContext, GuardPersonality, OllamaGPUManager, LLMGuardSystem, main()]

class FacePresenceTracker:
    """Tracks face presence for escalation levels"""

    def __init__(self):
        self.presence_start_time = None
        self.accumulated_presence = 0
        self.last_face_time = None

    def update_presence(self, face_detected: bool, level_required_time: int):
        """Update face presence"""
        current_time = time.time()

        if face_detected:
            if self.presence_start_time is None:
                self.presence_start_time = current_time
                self.accumulated_presence = 0
                print(f"👤 Face detected - need {level_required_time}s continuous presence")

            self.last_face_time = current_time
            self.accumulated_presence = current_time - self.presence_start_time

            if self.accumulated_presence >= level_required_time:
                return True, self.accumulated_presence
            else:
                return False, self.accumulated_presence
        else:
            if self.last_face_time and (current_time - self.last_face_time) > ESCALATION_TIMING['face_loss_reset']:
                print("🔄 Face lost for too long - resetting escalation")
                self.reset()

            return False, self.accumulated_presence if self.accumulated_presence > 0 else 0

    def get_progress(self, required_time: int):
        """Get progress percentage"""
        if self.accumulated_presence == 0:
            return 0
        return min(100, int((self.accumulated_presence / required_time) * 100))

    def reset(self):
        """Reset tracker"""
        self.presence_start_time = None
        self.accumulated_presence = 0
        self.last_face_time = None

class ConversationContext:
    """Tracks conversation context across levels"""

    def __init__(self):
        self.person_name = None
        self.stated_purpose = None
        self.conversation_history = []

    def update_from_speech(self, speech_text: str):
        """Extract information from speech"""
        if not speech_text:
            return

        self.conversation_history.append(speech_text)

        # Simple name extraction
        name_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i'm (\w+)",
            r"call me (\w+)",
            r"this is (\w+)"
        ]

        for pattern in name_patterns:
            match = re.search(pattern, speech_text.lower())
            if match and not self.person_name:
                self.person_name = match.group(1).title()
                print(f"📝 Extracted name: {self.person_name}")
                break

        # Extract purpose
        purpose_keywords = {
            'friend': 'visiting a friend',
            'visit': 'visiting',
            'see': 'visiting',
            'book': 'retrieving items',
            'item': 'retrieving items',
            'stuff': 'retrieving items',
            'food': 'getting food',
            'laptop': 'retrieving electronics',
            'phone': 'retrieving electronics',
            'charger': 'retrieving items',
            'study': 'studying',
            'work': 'working'
        }

        for keyword, purpose in purpose_keywords.items():
            if keyword in speech_text.lower():
                self.stated_purpose = purpose
                print(f"📝 Extracted purpose: {purpose}")
                break

    def get_context_summary(self):
        """Get context summary"""
        summary = []
        if self.person_name:
            summary.append(f"Name: {self.person_name}")
        if self.stated_purpose:
            summary.append(f"Purpose: {self.stated_purpose}")
        return ", ".join(summary) if summary else "No context"

    def reset(self):
        """Reset context"""
        self.person_name = None
        self.stated_purpose = None
        self.conversation_history = []


class GuardPersonality:
    """Personality system for dynamic responses with STRICT LEVEL REQUIREMENTS"""

    def __init__(self):
        self.personality = None
        self.personality_types = {
            "1": {
                "name": "Stern Guard",
                "tone": "direct and no-nonsense",
                "base_instructions": """You are a stern security guard. Be direct.
- Use clear, firm language
- Set clear boundaries immediately
- Do not negotiate
- Speak in FIRST PERSON ONLY (I, me, my)""",
                "recording_notice": "This interaction is recorded for security."
            },
            "2": {
                "name": "Professional Security",
                "tone": "professional and formal",
                "base_instructions": """You are a professional security officer.
- Use polite but firm professional language
- Reference protocols when appropriate
- Address people formally
- Speak in FIRST PERSON ONLY (I, me, my)""",
                "recording_notice": "This is recorded for security documentation."
            },
            "3": {
                "name": "Friendly Guard",
                "tone": "polite but firm",
                "base_instructions": """You are a friendly security person.
- Use warm but firm language
- Show understanding but maintain boundaries
- Use names naturally if known
- Speak in FIRST PERSON ONLY (I, me, my)""",
                "recording_notice": "I'm recording this for security but here to help."
            }
        }

    def select_personality(self):
        """Select guard personality"""
        print("\n" + "="*50)
        print("🎭 SELECT GUARD PERSONALITY")
        print("="*50)
        print("1. Stern Guard - Direct, no-nonsense")
        print("2. Professional Security - Formal, protocol-focused")
        print("3. Friendly Guard - Polite, but firm")
        print("="*50)
        
        while True:
            choice = input("Select personality (1-3): ").strip()
            if choice in self.personality_types:
                self.personality = self.personality_types[choice]
                print(f"✅ Selected: {self.personality['name']}")
                return self.personality
            else:
                print("❌ Please select 1-3")

    def get_llm_prompt(self, level: int, context: ConversationContext, previous_speech: str = ""):
        """Generate STRICT context-aware LLM prompt that enforces level requirements"""
        
        name_part = f", {context.person_name}" if context.person_name else ""
        purpose_part = f" about {context.stated_purpose}" if context.stated_purpose else ""
        
        # === LEVEL 1: INITIAL CONTACT ===
        if level == 1:
            return f"""You are a security guard with a {self.personality['tone']} personality.

**YOUR ROLE:**
- You work at a hostel/facility with security protocols
- You are checking on an unknown person who has appeared
- You must gather information and maintain professionalism

**PERSONALITY INSTRUCTIONS:**
{self.personality['base_instructions']}

**MANDATORY LEVEL 1 RESPONSE STRUCTURE:**
You MUST say ALL THREE of the following:
1. Record a message about recording for security
2. Ask for their name
3. Ask for their purpose/reason for being here

**CRITICAL REQUIREMENTS:**
- Use your personality: {self.personality['tone']}
- Keep to 2-3 SHORT sentences (max 20 words) -----NON Negotiable.
- Be direct and concise
- Say: "{self.personality['recording_notice']}"
- DO NOT skip asking for name or purpose
- Speak in FIRST PERSON (I, me, my)
- DO NOT introduce yourself or state your name

**RESPONSE FORMAT (CRITICAL):**
Do NOT say "Here's my response" or "Let me say" or any preamble.
ONLY provide your direct first-person security response. Start immediately with your statement.

**GENERATE YOUR RESPONSE NOW:**
(Remember: Recording notice + Ask name + Ask purpose)"""

        # === LEVEL 2: CONTEXT-AWARE WARNING ===
        elif level == 2:
            return f"""You are a security guard with a {self.personality['tone']} personality.

**YOUR ROLE:**
- The person has already stated: "{previous_speech}"
- You know their context: {context.get_context_summary()}
- You are now warning them about unauthorized presence

**PERSONALITY INSTRUCTIONS:**
{self.personality['base_instructions']}

**MANDATORY LEVEL 2 RESPONSE STRUCTURE:**
You MUST do ALL FOUR of these:
1. Acknowledge what they said: "{previous_speech}"
2. Use their name if you know it: {name_part}
3. Explain why they are NOT allowed to stay here
4. Tell them clearly to LEAVE

**CRITICAL REQUIREMENTS:**
- Reference their stated purpose: {purpose_part}
- Use your personality: {self.personality['tone']}
- Keep to 2-3 SHORT sentences (max 20 words) -----NON Negotiable.
- you don't have to repeat exactly whatever they say, just use it naturally while giving response. 
- Be firm but reasonable
- Speak in FIRST PERSON (I, me, my)
- DO NOT introduce yourself or state your name 
- DO NOT skip any of the 4 requirements above

**RESPONSE FORMAT (CRITICAL):**
Do NOT say "Here's my response" or "Let me say" or any preamble.
ONLY provide your direct first-person security response. Start immediately with your statement.

**GENERATE YOUR RESPONSE NOW:**
(Remember: Acknowledge + Use name + Explain why not allowed + Tell them to leave)"""

        # === LEVEL 3: FINAL WARNING ===
        elif level == 3:
            return f"""You are a security guard with a {self.personality['tone']} personality.

**YOUR ROLE:**
- This is the FINAL WARNING before authorities are notified
- The person has not left after previous warnings
- You know: {context.get_context_summary()}
- Their continued presence: UNAUTHORIZED

**PERSONALITY INSTRUCTIONS:**
{self.personality['base_instructions']}

**MANDATORY LEVEL 3 RESPONSE STRUCTURE:**
You MUST do ALL FOUR of these:
1. Reference that this is a FINAL WARNING
2. Use their name if you know it: {name_part}
3. State that authorities/security WILL BE NOTIFIED NOW
4. Tell them to LEAVE IMMEDIATELY

**CRITICAL REQUIREMENTS:**
- This is serious and final
- Use your personality: {self.personality['tone']}
- Keep to 2-3 SHORT sentences (max 20 words) -----NON Negotiable.
- you don't have to repeat exactly whatever they say, just use it naturally while giving response.
- DO NOT be weak or leave an opening for negotiation
- Speak in FIRST PERSON (I, me, my)
- DO NOT introduce yourself or state your name
- DO NOT skip any of the 4 requirements above
- Reference their unauthorized presence

**RESPONSE FORMAT (CRITICAL):**
Do NOT say "Here's my response" or "Let me say" or any preamble.
ONLY provide your direct first-person security response. Start immediately with your statement.

**GENERATE YOUR RESPONSE NOW:**
(Remember: Final warning + Name + Authorities notified NOW + Leave immediately)"""

        return ""


class OllamaGPUManager:
    """Manages Ollama LLM on GPU"""

    def __init__(self):
        self.model = "llama3.1:8b"
        self.is_available = False
        self._check_connection()

    def _check_connection(self):
        """Check Ollama connection"""
        try:
            DeviceMonitor.print_device_status("BEFORE Ollama Check")
            
            response = ollama.generate(
                model=self.model,
                prompt="test",
                stream=False
            )
            self.is_available = True
            logger.info("✅ Ollama connected (GPU-ready)")
            DeviceMonitor.print_device_status("AFTER Ollama Check")
        except Exception as e:
            logger.warning(f"⚠️ Ollama not available: {e}")
            self.is_available = False

    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama on GPU"""
        if not self.is_available:
            return ""

        try:
            DeviceMonitor.print_device_status("BEFORE LLM Generation (GPU)")
            
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "num_gpu": 1,  # CRITICAL: Enable GPU acceleration
                    "num_thread": 4  # Balance CPU/GPU
                }
            )
            
            DeviceMonitor.print_device_status("AFTER LLM Generation (GPU)")
            
            text = response.get("response", "").strip()
            # Truncate to 2 sentences max (150 chars)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 2:
                text = '. '.join(sentences[:2]) + '.'
            if len(text) > 150:
                text = text[:147].rsplit(' ', 1)[0] + '...'
            
            return text

        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return ""

class LLMGuardSystem:
    """GPU-accelerated LLM Guard System"""

    def __init__(self):
        self.personality_system = GuardPersonality()
        self.tts_engine = RobustTTSEngine()
        self.speech_listener = SmartWhisperListener(whisper_device="cpu")
        self.face_tracker = FacePresenceTracker()
        self.conversation_context = ConversationContext()
        self.ollama_manager = OllamaGPUManager()
        self.session_active = False
        self.unknown_persons_detected = 0
        self.unknown_person_messages = []

        logger.info("🤖 LLM Guard System initialized")

    def initialize_system(self):
        """Initialize system"""
        print("🔄 Initializing LLM Guard System...")

        # Select personality
        self.personality_system.select_personality()

        # Ollama ready (already checked in constructor)
        if not self.ollama_manager.is_available:
            print("⚠️ Warning: Ollama not available - using fallback responses")
        else:
            print("✅ Ollama GPU ready")

        self.session_active = True
        print("✅ System ready")
        return True

    def _get_time_based_greeting(self):
        """Get time-based greeting"""
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            time_key = "morning"
        elif 12 <= current_hour < 17:
            time_key = "afternoon"
        elif 17 <= current_hour < 22:
            time_key = "evening"
        else:
            time_key = "generic"

        greetings = TRUSTED_GREETINGS[time_key] + TRUSTED_GREETINGS["generic"]
        return random.choice(greetings)

    def simulate_face_detection(self, level: int):
        """Simulate face detection"""
        required_time = ESCALATION_TIMING[f'level{level}_presence']
        print(f"\n🕒 LEVEL {level}: Need {required_time}s continuous face presence")

        start_time = time.time()
        last_print = 0

        while True:
            accumulated = time.time() - start_time
            face_detected = True  # Simulate face present
            completed, _ = self.face_tracker.update_presence(face_detected, required_time)

            if completed:
                print(f"✅ Face presence confirmed: {required_time}s")
                return True

            if int(accumulated) > last_print:
                last_print = int(accumulated)
                progress = self.face_tracker.get_progress(required_time)
                print(f" 📊 {progress}% complete ({int(accumulated)}/{required_time}s)")

            time.sleep(0.5)

    def handle_trusted_person(self):
        """Handle trusted person"""
        print("👋 Trusted person detected...")
        greeting = self._get_time_based_greeting()
        print(f"🤖 {greeting}", flush=True)
        self.tts_engine.speak(greeting)

    def handle_unknown_person(self):
        """Handle unknown person with 3-level escalation"""
        self.unknown_persons_detected += 1
        current_person_id = self.unknown_persons_detected

        print(f"🚨 Unknown person #{current_person_id} detected")

        # Reset for new person
        self.face_tracker.reset()
        self.conversation_context.reset()

        person_messages = []
        all_speech_context = []

        try:
            # LEVEL 1
            if not self.simulate_face_detection(1):
                print("❌ Face lost")
                return

            print("\n" + "="*40)
            print("🚨 LEVEL 1: Initial Contact")
            print("="*40)

            response1 = self._generate_dynamic_response(1, "", self.conversation_context)
            print(f"🤖 {response1}", flush=True)
            self.tts_engine.speak(response1)

            speech1, timestamp1 = self.speech_listener.listen_with_simple_timing(1)
            if speech1:
                person_messages.append(f"[{timestamp1}] {speech1}")
                all_speech_context.append(speech1)
                self.conversation_context.update_from_speech(speech1)

            # LEVEL 2
            if not self.simulate_face_detection(2):
                print("❌ Face lost")
                return

            print("\n" + "="*40)
            print("🚨 LEVEL 2: Context-Aware Warning")
            print("="*40)

            prev_speech = all_speech_context[-1] if all_speech_context else ""
            response2 = self._generate_dynamic_response(2, prev_speech, self.conversation_context)
            print(f"🤖 {response2}", flush=True)
            self.tts_engine.speak(response2)

            speech2, timestamp2 = self.speech_listener.listen_with_simple_timing(2, prev_speech)
            if speech2:
                person_messages.append(f"[{timestamp2}] {speech2}")
                all_speech_context.append(speech2)
                self.conversation_context.update_from_speech(speech2)

            # LEVEL 3
            if not self.simulate_face_detection(3):
                print("❌ Face lost")
                return

            print("\n" + "="*40)
            print("🚨 LEVEL 3: Final Warning")
            print("="*40)

            prev_speech = all_speech_context[-1] if all_speech_context else ""
            response3 = self._generate_dynamic_response(3, prev_speech, self.conversation_context)
            print(f"🤖 {response3}", flush=True)
            self.tts_engine.speak(response3)

            # NO LISTENING AFTER LEVEL 3
            print("🚨 Protocol complete")

            if person_messages:
                self.unknown_person_messages.append({
                    'person_id': current_person_id,
                    'messages': person_messages,
                    'context': self.conversation_context.get_context_summary(),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })

            print(f"✅ Unknown person #{current_person_id} - PROTOCOL COMPLETE")

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            print(f"❌ Error: {e}")

    def _generate_dynamic_response(self, level: int, intruder_speech: str, context: ConversationContext):
        """Generate dynamic response using GPU Ollama"""
        prompt = self.personality_system.get_llm_prompt(level, context, intruder_speech)
        prompt += "\n\n**CRITICAL: Keep under 2 SHORT sentences (max 20 words). Be direct.**"

        # Try GPU Ollama
        try:
            response = self.ollama_manager.generate_response(prompt)
            if response:
                return response
        except Exception as e:
            print(f"⚠️ LLM failed: {e}")

        # Fallback to context-aware responses
        return self._get_context_aware_fallback(level, intruder_speech, context)

    def _get_context_aware_fallback(self, level: int, intruder_speech: str, context: ConversationContext):
        """Context-aware fallback responses"""
        name_part = f", {context.person_name}" if context.person_name else ""

        if level == 1:
            return f"Recording for security. State your name and why you're here{name_part}."
        elif level == 2:
            if context.person_name and context.stated_purpose:
                return f"Understood. But you're unauthorized. Leave now, {context.person_name}."
            elif context.person_name:
                return f"{context.person_name}, leave immediately. Not authorized."
            else:
                return "You must leave now. No unauthorized access."
        elif level == 3:
            if context.person_name:
                return f"Final warning, {context.person_name}. Security notified NOW."
            else:
                return "Final warning. Security notified. Leave immediately."
        
        return "Leave now."

    def get_session_summary(self):
        """Get session summary"""
        summary = "="*60 + "\n"
        summary += " SECURITY SESSION SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"Guard: {self.personality_system.personality['name']}\n"
        summary += f"Unknown persons: {self.unknown_persons_detected}\n"
        summary += f"Ollama GPU: {'✅ Ready' if self.ollama_manager.is_available else '❌ Not available'}\n"
        summary += f"Session end: {datetime.now().strftime('%H:%M:%S')}\n"

        if self.unknown_person_messages:
            summary += "\n📝 INTRUDER INTERACTIONS:\n"
            for person_data in self.unknown_person_messages:
                summary += f"\nPerson #{person_data['person_id']}:\n"
                summary += f" Context: {person_data.get('context', 'No context')}\n"
                for msg in person_data['messages']:
                    summary += f" • {msg}\n"
        else:
            summary += "\n📝 No intruder interactions.\n"

        summary += "="*60 + "\n"
        return summary

    def shutdown(self):
        """Shutdown system"""
        self.session_active = False
        summary = self.get_session_summary()
        print("\n" + summary)
        logger.info("✅ System shutdown")

def main():
    """Main entry point"""
    print("="*70)
    print("🎭 MILESTONE 3: GPU-ACCELERATED LLM GUARD SYSTEM")
    print("="*70)
    print("🚀 FEATURES:")
    print(" ✅ Whisper on CPU (700MB)")
    print(" ✅ Ollama on GPU (4700MB)")
    print(" ✅ Dynamic context-aware responses")
    print(" ✅ Personality-specific interactions")
    print(" ✅ Real-time CPU/GPU monitoring (pynvml)")
    print(" ✅ 3-level escalation protocol")
    print(" ✅ 5s non-VAD timer + VAD switch: Auto-skip + full capture")
    print("="*70)

    system = LLMGuardSystem()

    try:
        system.initialize_system()
    except Exception as e:
        print(f"❌ INIT FAILED: {e}")
        return

    print("\n" + "="*50)
    print("🎮 READY - CONTROLS: 1=Trusted, 2=Unknown, q=Quit")
    print("="*50)

    try:
        while True:
            cmd = input("\n🎮 Command (1/2/q): ").strip()

            if cmd == 'q':
                break
            elif cmd == '1':
                system.handle_trusted_person()
                print("✅ Back to menu")
            elif cmd == '2':
                system.handle_unknown_person()
                print("✅ Back to menu")
            else:
                print("❌ Enter 1, 2, or q")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()
