"""
AI Guard Bot - FINAL VERSION WITH ALL FIXES
- Whisper parameters: logprob_threshold + condition_on_previous_text
- Pause audio during TTS (no echo)
- Stricter Whisper parameters (no hallucinations)
- Single-word filter (no junk)
"""

import os
import sys
import io
import time
import cv2
import torch
import whisper
import pyaudio
import pyttsx3
import ollama
import pickle
import numpy as np
import threading
import signal
from pathlib import Path
from datetime import datetime
from collections import deque
from enum import Enum
from difflib import SequenceMatcher

# ============================================================================
# WINDOWS UNICODE FIX
# ============================================================================
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, TypeError):
        pass

# ============================================================================
# GPU INITIALIZATION
# ============================================================================
print("="*70, flush=True)
print("🔧 INITIALIZING AI GUARD BOT", flush=True)
print("="*70, flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"✅ Dedicated VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)
else:
    print("⚠️  No GPU - using CPU", flush=True)
    sys.exit(1)

# ============================================================================
# IMPORTS
# ============================================================================
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    print("❌ facenet-pytorch not installed. Run: pip install facenet-pytorch", flush=True)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "embeddings" / "known_faces.pkl"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16

KNOWN_FACE_TOLERANCE = 0.40
MIN_CONFIDENCE_KNOWN = 0.75
DETECTION_INTERVAL = 2

ESCALATION_TIMING = {
    "level1_presence": 2,
    "level2_presence": 4,
    "level3_presence": 6,
    "face_loss_reset": 7,
    "cooldown_duration": 30
}

PERSONALITIES = {
    "1": {"name": "Stern Guard", "tone": "direct and no-nonsense",
          "recording_notice": "This interaction is recorded for security."},
    "2": {"name": "Professional Security", "tone": "professional and formal",
          "recording_notice": "This is recorded for security documentation."},
    "3": {"name": "Friendly Guard", "tone": "polite but firm",
          "recording_notice": "I'm recording this for security but here to help."}
}

ACTIVATION_PHRASES = [
    "guard mode", "activate guard", "start guard",
    "guard my room", "activate security", "start guarding"
]

DEACTIVATION_PHRASES = [
    "deactivate", "stop guard", "shutdown",
    "stand down", "guard off", "stop guarding"
]

SIMILARITY_THRESHOLD = 0.6

class State(Enum):
    IDLE = "idle"
    M1_VOICE = "m1_voice"
    M2_FACE = "m2_face"
    M3_ESCALATION = "m3_escalation"

# ============================================================================
# HELPER CLASSES
# ============================================================================

class AccumulatedTimer:
    """Timer with pause/resume/reset logic"""
    def __init__(self, target_seconds):
        self.target = target_seconds
        self.accumulated = 0.0
        self.last_seen_time = None
        self.is_active = False
        self.lock = threading.Lock()
    
    def update(self, face_present):
        with self.lock:
            current_time = time.time()
            
            if face_present:
                if not self.is_active:
                    self.is_active = True
                    self.last_seen_time = current_time
                else:
                    delta = current_time - self.last_seen_time
                    self.accumulated += delta
                    self.last_seen_time = current_time
            else:
                if self.is_active:
                    time_gone = current_time - self.last_seen_time
                    if time_gone > ESCALATION_TIMING["face_loss_reset"]:
                        print(f"⚠️  Unknown gone >{ESCALATION_TIMING['face_loss_reset']}s - Timer reset", flush=True)
                        self.reset()
            
            return self.accumulated >= self.target
    
    def reset(self):
        with self.lock:
            self.accumulated = 0.0
            self.last_seen_time = None
            self.is_active = False


class ConversationContext:
    """Track conversation across escalation levels"""
    def __init__(self):
        self.person_name = None
        self.stated_purpose = None
        self.conversation_history = []
        self.lock = threading.Lock()
    
    def update_from_speech(self, speech_text):
        with self.lock:
            if not speech_text:
                return
            
            self.conversation_history.append(speech_text)
            speech_lower = speech_text.lower()
            
            import re
            name_patterns = [r"my name is (\w+)", r"i am (\w+)", r"i'm (\w+)", r"call me (\w+)"]
            for pattern in name_patterns:
                match = re.search(pattern, speech_lower)
                if match and not self.person_name:
                    self.person_name = match.group(1).title()
                    print(f"   📝 Extracted name: {self.person_name}", flush=True)
                    break
            
            purpose_keywords = {
                "friend": "visiting a friend", "visit": "visiting",
                "book": "retrieving items", "laptop": "retrieving electronics",
                "study": "studying"
            }
            for keyword, purpose in purpose_keywords.items():
                if keyword in speech_lower and not self.stated_purpose:
                    self.stated_purpose = purpose
                    print(f"   📝 Extracted purpose: {purpose}", flush=True)
                    break
    
    def get_context_summary(self):
        with self.lock:
            parts = []
            if self.person_name:
                parts.append(f"Name: {self.person_name}")
            if self.stated_purpose:
                parts.append(f"Purpose: {self.stated_purpose}")
            return ", ".join(parts) if parts else "No context"
    
    def reset(self):
        with self.lock:
            self.person_name = None
            self.stated_purpose = None
            self.conversation_history = []


# ============================================================================
# SIMILARITY CALCULATION
# ============================================================================

def calculate_similarity(text, phrases):
    """Calculate similarity between transcribed text and command phrases"""
    if not text or not text.strip():
        return 0.0, ""
    
    text = text.lower().strip()
    best_similarity = 0.0
    best_phrase = ""
    
    for phrase in phrases:
        seq_similarity = SequenceMatcher(None, text, phrase).ratio()
        
        text_words = set(text.split())
        phrase_words = set(phrase.split())
        word_similarity = len(text_words.intersection(phrase_words)) / max(len(phrase_words), 1)
        
        combined_score = 0.7 * seq_similarity + 0.3 * word_similarity
        
        if combined_score > best_similarity:
            best_similarity = combined_score
            best_phrase = phrase
    
    return best_similarity, best_phrase


# ============================================================================
# CAMERA THREAD
# ============================================================================

class CameraThread(threading.Thread):
    """Dedicated camera thread - 20 FPS guaranteed"""
    
    def __init__(self, camera_id=0):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        self.current_frame = None
        self.display_frame = None
        self.faces_data = []
        self.fps = 0
        self.lock = threading.Lock()
    
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("❌ Camera failed to open", flush=True)
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        fps_counter = 0
        fps_start = time.time()
        
        print("📹 Camera thread started (20 FPS)", flush=True)
        
        while self.running:
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            with self.lock:
                self.current_frame = frame.copy()
            
            display = self.draw_overlays(frame)
            
            with self.lock:
                self.display_frame = display
            
            cv2.imshow("AI Guard Bot", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.running = False
                break
            
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, 0.05 - elapsed))
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("📹 Camera thread stopped", flush=True)
    
    def draw_overlays(self, frame):
        display = frame.copy()
        
        with self.lock:
            faces = self.faces_data.copy()
        
        for face in faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']
            
            if name != "Unknown" and confidence >= MIN_CONFIDENCE_KNOWN:
                color = (0, 255, 0)
                label = f"{name} ({confidence:.2f})"
            else:
                color = (0, 0, 255)
                label = f"Unknown ({confidence:.2f})"
            
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(display, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(display, f"FPS: {self.fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, timestamp, (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display
    
    def get_current_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def update_faces_data(self, faces):
        with self.lock:
            self.faces_data = faces
    
    def stop(self):
        self.running = False


# ============================================================================
# FACE PROCESSING THREAD
# ============================================================================

class FaceProcessingThread(threading.Thread):
    """Async face detection and recognition"""
    
    def __init__(self, camera_thread, face_encoder, known_encodings, known_names):
        super().__init__(daemon=True)
        self.camera_thread = camera_thread
        self.face_encoder = face_encoder
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.running = False
        self.device = device
        self.frame_counter = 0
    
    def run(self):
        self.running = True
        print("👤 Face processing thread started", flush=True)
        
        while self.running:
            frame = self.camera_thread.get_current_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            self.frame_counter += 1
            if self.frame_counter % DETECTION_INTERVAL != 0:
                time.sleep(0.05)
                continue
            
            faces_data = self.process_frame(frame)
            self.camera_thread.update_faces_data(faces_data)
            
            time.sleep(0.05)
        
        print("👤 Face processing thread stopped", flush=True)
    
    def process_frame(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return []
        
        face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
        encodings = self.encode_faces(frame, face_locations)
        
        results = []
        for location, encoding in zip(face_locations, encodings):
            name, confidence = self.match_face(encoding)
            results.append({
                'location': location,
                'name': name,
                'confidence': confidence
            })
        
        return results
    
    def encode_faces(self, frame, face_locations):
        encodings = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            for (top, right, bottom, left) in face_locations:
                face_image = rgb_frame[top:bottom, left:right]
                if face_image.size == 0:
                    continue
                
                face_resized = cv2.resize(face_image, (160, 160))
                face_tensor = torch.tensor(face_resized, dtype=torch.float32)
                face_tensor = face_tensor.permute(2, 0, 1)
                face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                embedding = self.face_encoder(face_tensor)
                encodings.append(embedding.squeeze(0))
        
        return encodings
    
    def match_face(self, face_encoding):
        if not self.known_encodings:
            return "Unknown", 0.0
        
        best_match = None
        best_distance = float('inf')
        
        with torch.no_grad():
            face_encoding = face_encoding.unsqueeze(0)
            
            for i, known_encoding in enumerate(self.known_encodings):
                similarity = torch.nn.functional.cosine_similarity(
                    face_encoding,
                    known_encoding.unsqueeze(0)
                )
                distance = 1.0 - similarity.item()
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = i
        
        if best_match is not None and best_distance < KNOWN_FACE_TOLERANCE:
            name = self.known_names[best_match]
            confidence = 1.0 - best_distance
            return name, float(confidence)
        else:
            return "Unknown", float(1.0 - best_distance) if best_match is not None else 0.0
    
    def stop(self):
        self.running = False


# ============================================================================
# AUDIO THREAD (ALL FIXES APPLIED)
# ============================================================================

class AudioThread(threading.Thread):
    """Background audio - all fixes applied"""
    
    def __init__(self, whisper_model, parent_bot):
        super().__init__(daemon=True)
        self.whisper_model = whisper_model
        self.parent_bot = parent_bot
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.listening_active = False
        self.last_transcription_time = 0
        self.command_cooldown = 3.0
        self.lock = threading.Lock()
        
        self.deactivation_buffer = deque(maxlen=5)
    
    def run(self):
        self.running = True
        print("🎤 Audio thread started", flush=True)
        
        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        while self.running:
            if not self.listening_active:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            if current_time - self.last_transcription_time < self.command_cooldown:
                time.sleep(0.1)
                continue
            
            transcription = self.transcribe_audio_chunk(duration=3)
            
            if transcription and len(transcription.strip()) >= 3:
                # ✅ FIX 4: Filter single-word hallucinations
                words = transcription.strip().split()
                if len(words) < 2:
                    continue
                
                print(f"   💬 Heard: '{transcription}'", flush=True)
                
                self.parent_bot.process_transcription(transcription)
                
                self.last_transcription_time = current_time
                
                with self.lock:
                    self.deactivation_buffer.append(transcription.lower())
            
            time.sleep(0.1)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🎤 Audio thread stopped", flush=True)
    
    def transcribe_audio_chunk(self, duration=3):
        """✅ FIX 1 & 3: Whisper parameters + stricter thresholds"""
        frames = []
        chunks_needed = int(SAMPLE_RATE / CHUNK_SIZE * duration)
        
        for _ in range(chunks_needed):
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except:
                break
        
        if not frames:
            return ""
        
        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0
        
        try:
            result = self.whisper_model.transcribe(
                audio_np,
                language="en",
                fp16=True,
                no_speech_threshold=0.8,             # ✅ FIX 3: Stricter (was 0.7)
                logprob_threshold=-0.5,              # ✅ FIX 1: Require higher confidence
                condition_on_previous_text=False,    # ✅ FIX 1: No context bias
                temperature=0.0,                     # ✅ FIX 3: Deterministic
                compression_ratio_threshold=2.4      # ✅ FIX 3: Reject repetitive
            )
            return result["text"].strip()
        except:
            return ""
    
    def get_deactivation_commands(self):
        with self.lock:
            return list(self.deactivation_buffer)
    
    def clear_deactivation_buffer(self):
        with self.lock:
            self.deactivation_buffer.clear()
    
    def start_listening(self):
        with self.lock:
            self.listening_active = True
    
    def stop_listening(self):
        with self.lock:
            self.listening_active = False
    
    def stop(self):
        self.running = False


# ============================================================================
# MAIN STATE MACHINE
# ============================================================================

class GuardBotStateMachine:
    
    def __init__(self):
        print("\n" + "="*70, flush=True)
        print("🤖 LOADING ALL MODELS (OPTIMIZED ORDER)", flush=True)
        print("="*70, flush=True)
        print("⚙️  Strategy: Ollama → FaceNet → Whisper (GPU)", flush=True)
        print("="*70, flush=True)
        
        self.device = device
        self.state = State.IDLE
        self.personality = None
        self.shutdown_requested = False
        
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Load Ollama FIRST
        print("\n📥 [1/3] Connecting to Ollama (Llama3.1:8b) - 4.7GB...", flush=True)
        start = time.time()
        try:
            ollama.generate(model='llama3.1:8b', prompt='Initialize', stream=False)
            print(f"✅ Ollama loaded in {time.time()-start:.1f}s", flush=True)
        except Exception as e:
            print(f"❌ Ollama failed: {e}", flush=True)
            print("   Run: ollama serve", flush=True)
            sys.exit(1)
        
        # Load FaceNet
        print("\n📥 [2/3] Loading FaceNet InceptionResnetV1 (GPU) - 200MB...", flush=True)
        start = time.time()
        self.face_encoder = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        torch.backends.cudnn.benchmark = True
        print(f"✅ FaceNet loaded in {time.time()-start:.1f}s", flush=True)
        
        # Load Whisper Medium GPU
        print("\n📥 [3/3] Loading Whisper Medium (GPU) - 1.5GB...", flush=True)
        print("   💡 Loading after Ollama to avoid CUDA conflicts", flush=True)
        start = time.time()
        self.whisper_model = whisper.load_model("medium", device=self.device)
        print(f"✅ Whisper loaded in {time.time()-start:.1f}s", flush=True)
        print(f"   Expected latency: ~0.5s per 3s audio (GPU)", flush=True)
        
        # Load known faces
        print("\n📥 Loading known face embeddings...", flush=True)
        self.known_face_encodings = []
        self.known_face_names = []
        
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                
                if 'encodings' in data:
                    encodings_np = data['encodings']
                    self.known_face_names = data['names']
                elif 'face_encodings' in data:
                    encodings_np = data['face_encodings']
                    self.known_face_names = data['face_names']
                else:
                    encodings_np = []
                
                for enc_np in encodings_np:
                    enc_tensor = torch.tensor(enc_np, dtype=torch.float32).to(self.device)
                    self.known_face_encodings.append(enc_tensor)
                
                print(f"✅ Loaded {len(self.known_face_names)} known faces", flush=True)
                for name in set(self.known_face_names):
                    count = self.known_face_names.count(name)
                    print(f"   👤 {name}: {count} encodings", flush=True)
        else:
            print(f"⚠️  No embeddings found at: {EMBEDDINGS_FILE}", flush=True)
        
        self.camera_thread = None
        self.face_thread = None
        self.audio_thread = None
        
        self.unknown_timer = AccumulatedTimer(ESCALATION_TIMING["level1_presence"])
        self.cooldown_active = False
        self.cooldown_start = 0
        
        self.conversation_context = ConversationContext()
        
        print("\n" + "="*70, flush=True)
        print("✅ ALL MODELS LOADED SUCCESSFULLY", flush=True)
        print("="*70, flush=True)
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)
            print(f"   Ollama: ~4.7GB", flush=True)
            print(f"   FaceNet: ~0.2GB", flush=True)
            print(f"   Whisper: ~1.5GB", flush=True)
            print(f"   Total: ~6.4GB (100% optimal)", flush=True)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Ctrl+C detected - Shutting down...", flush=True)
        self.shutdown_requested = True
        self.cleanup()
    
    def speak(self, text):
        if not text:
            return
        
        print(f"\n🔊 GUARD: {text}", flush=True)
        
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.setProperty('volume', 0.95)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️  TTS error: {e}", flush=True)
    
    def process_transcription(self, transcription):
        """Process transcription immediately (called from audio thread)"""
        if self.state != State.M1_VOICE:
            return
        
        similarity, matched_phrase = calculate_similarity(transcription, ACTIVATION_PHRASES)
        
        if similarity >= SIMILARITY_THRESHOLD:
            print(f"   🎯 ACTIVATION DETECTED! (Similarity: {similarity:.2f}, Matched: '{matched_phrase}')", flush=True)
            self.activate_guard()
    
    def activate_guard(self):
        """✅ FIX 2: Pause audio before TTS"""
        # Pause audio thread before TTS
        self.audio_thread.stop_listening()
        time.sleep(0.3)  # Let in-flight transcription finish
        
        self.speak("Guard mode activated")
        
        # Clear buffer after TTS
        self.audio_thread.clear_deactivation_buffer()
        
        # Start camera/face threads
        self.camera_thread = CameraThread(camera_id=0)
        self.camera_thread.start()
        time.sleep(0.5)
        
        self.face_thread = FaceProcessingThread(
            self.camera_thread,
            self.face_encoder,
            self.known_face_encodings,
            self.known_face_names
        )
        self.face_thread.start()
        
        self.state = State.M2_FACE
        print("📹 Camera active - Face monitoring started", flush=True)
        
        # Resume audio thread AFTER everything ready
        self.audio_thread.start_listening()
    
    def listen_with_silence_detection(self, countdown=5, max_duration=10):
        """✅ FIX 1 & 3: Stricter Whisper parameters for M3"""
        print(f"\n🎤 LISTENING: {countdown}s countdown, then until silence (max {max_duration}s)", flush=True)
        
        ENERGY_THRESHOLD = 500
        SILENCE_TIMEOUT = 2.0
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        audio_buffer = []
        speech_detected = False
        
        try:
            print("   Phase 1: Countdown", end="", flush=True)
            for sec in range(countdown, 0, -1):
                print(f" {sec}...", end="", flush=True)
                sec_start = time.time()
                
                while time.time() - sec_start < 1.0:
                    try:
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        chunk = np.frombuffer(data, dtype=np.int16)
                        audio_buffer.append(chunk)
                        
                        if not speech_detected and np.max(np.abs(chunk)) > ENERGY_THRESHOLD:
                            speech_detected = True
                            print("\n   ✅ Speech detected!", flush=True)
                    except:
                        pass
            
            print("\n   Phase 1 complete", flush=True)
            
            if not speech_detected:
                print("   ⏭️  AUTO-SKIP: No speech", flush=True)
                return ""
            
            print(f"   Phase 2: Until {SILENCE_TIMEOUT}s silence...", flush=True)
            phase2_audio = []
            last_speech_time = time.time()
            phase2_start = time.time()
            
            while time.time() - phase2_start < max_duration:
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    chunk = np.frombuffer(data, dtype=np.int16)
                    phase2_audio.append(chunk)
                    
                    if np.max(np.abs(chunk)) > ENERGY_THRESHOLD:
                        last_speech_time = time.time()
                    
                    if time.time() - last_speech_time >= SILENCE_TIMEOUT:
                        print(f"   ✅ Silence detected", flush=True)
                        break
                except:
                    break
            
            if phase2_audio:
                combined = np.concatenate([np.concatenate(audio_buffer), np.concatenate(phase2_audio)])
            else:
                combined = np.concatenate(audio_buffer)
            
            print(f"   📊 Audio: {len(combined)/SAMPLE_RATE:.2f}s", flush=True)
            print(f"   ⏳ Transcribing (GPU Whisper)...", flush=True)
            
            audio_float = combined.astype(np.float32) / 32768.0
            result = self.whisper_model.transcribe(
                audio_float, 
                language="en", 
                fp16=True,
                no_speech_threshold=0.8,             # ✅ FIX 3: Stricter
                logprob_threshold=-0.5,              # ✅ FIX 1: Higher confidence
                condition_on_previous_text=False,    # ✅ FIX 1: No bias
                temperature=0.0,                     # ✅ FIX 3: Deterministic
                compression_ratio_threshold=2.4      # ✅ FIX 3: Reject repetitive
            )
            transcription = result["text"].strip()
            
            if transcription:
                print(f"\n   🗣️  INTRUDER: {transcription}", flush=True)
            else:
                print("   ⚠️  No clear speech", flush=True)
            
            return transcription
            
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            time.sleep(0.2)
    
    def generate_llm_response(self, level, previous_speech, context):
        """
        Generate context-aware guard responses with Jarvis identity
        - Max 20 words enforced
        - No name hallucination
        - Natural flow prompts
        """
        
        # ✅ Jarvis identity
        jarvis_intro = "You are Jarvis, an AI security assistant."
        
        # ✅ Anti-hallucination name handling
        if context.person_name:
            name_instruction = f"Use their name: {context.person_name}"
            name_example = f"Good: '{context.person_name}, you need to leave.'"
        else:
            name_instruction = "CRITICAL: No name given. Say 'you' or 'sir/ma'am' ONLY. Do NOT invent names."
            name_example = "Good: 'You need to leave now.' | Bad: 'John, leave now.' (hallucinated name)"
        
        # Level 1: Initial Contact
        if level == 1:
            prompt = f"""{jarvis_intro}
    You have a {self.personality['tone']} personality.

    SITUATION:
    An unknown person appeared. You must verify their identity.

    WHAT TO SAY (naturally cover these points):
    - Inform them: "{self.personality['recording_notice']}"
    - Ask who they are
    - Ask why they're here

    {name_instruction}

    CRITICAL CONSTRAINTS:
    - Maximum 20 words total (count them!)
    - First person only (I, me, my)
    - Start speaking immediately (no "Here's my response...")
    - Be {self.personality['tone']}

    EXAMPLE GOOD RESPONSES:
    "{self.personality['recording_notice']} Who are you and why are you here?"
    "I'm recording for security. State your name and purpose please."

    Your response (20 words max):"""

        # Level 2: Context-Aware Warning
        elif level == 2:
            context_info = context.get_context_summary()
            
            prompt = f"""{jarvis_intro}
    You have a {self.personality['tone']} personality.

    SITUATION:
    They said: "{previous_speech}"
    You know: {context_info}
    They are NOT authorized. They must leave.

    WHAT TO SAY (naturally cover these points):
    - Acknowledge what they told you
    - {name_instruction}
    - Explain they can't stay
    - Tell them to leave now

    CRITICAL CONSTRAINTS:
    - Maximum 20 words total (count them!)
    - First person only (I, me, my)
    - If no name, use 'you' not made-up names
    - Be {self.personality['tone']}

    {name_example}

    Your response (20 words max):"""

        # Level 3: Final Warning
        else:
            context_info = context.get_context_summary()
            
            prompt = f"""{jarvis_intro}
    You have a {self.personality['tone']} personality.

    SITUATION:
    FINAL WARNING. They ignored previous warnings.
    Known: {context_info}

    WHAT TO SAY (naturally cover these points):
    - State "FINAL WARNING"
    - {name_instruction}
    - Say authorities are being notified RIGHT NOW
    - Command them to leave IMMEDIATELY

    CRITICAL CONSTRAINTS:
    - Maximum 20 words total (count them!)
    - First person only (I, me, my)
    - Serious and authoritative tone
    - NO made-up names
    - Start immediately

    {name_example}

    Your response (20 words max):"""
        
        # ✅ Generate with Ollama
        try:
            response = ollama.generate(
                model='llama3.1:8b',
                prompt=prompt,
                stream=False
            )
            text = response.get('response', '').strip()
            
            # ✅ Post-processing: enforce 20-word limit
            words = text.split()
            if len(words) > 20:
                text = ' '.join(words[:20])
                if not text.endswith('.'):
                    text += '.'
            
            # ✅ Remove meta-commentary if LLM adds it
            if text.lower().startswith(("here's", "here is", "my response")):
                sentences = text.split('.')
                if len(sentences) > 1:
                    text = '.'.join(sentences[1:]).strip()
            
            # ✅ Ensure it ends with punctuation
            if text and not text[-1] in '.!?':
                text += '.'
            
            return text
        
        except Exception as e:
            print(f"⚠️  LLM error: {e}", flush=True)
            
            # ✅ Fallback responses (Jarvis-style, no name hallucination)
            if level == 1:
                return f"{self.personality['recording_notice']} Who are you? Why are you here?"
            elif level == 2:
                if context.person_name:
                    return f"{context.person_name}, you're not authorized. Leave now."
                else:
                    return "You're not authorized here. You need to leave immediately."
            else:
                if context.person_name:
                    return f"FINAL WARNING {context.person_name}. Security alerted. Leave now."
                else:
                    return "FINAL WARNING. Authorities notified. Leave immediately."


    def handle_idle(self):
        print("\n" + "="*70, flush=True)
        print("🎭 SELECT GUARD PERSONALITY", flush=True)
        print("="*70, flush=True)
        print("1. Stern Guard - Direct, no-nonsense", flush=True)
        print("2. Professional Security - Formal, protocol-focused", flush=True)
        print("3. Friendly Guard - Polite, but firm", flush=True)
        print("="*70, flush=True)
        
        while True:
            choice = input("Select personality (1-3): ").strip()
            if choice in PERSONALITIES:
                self.personality = PERSONALITIES[choice]
                print(f"✅ Selected: {self.personality['name']}", flush=True)
                break
            else:
                print("❌ Select 1, 2, or 3", flush=True)
        
        self.state = State.M1_VOICE
        print("\n" + "="*70, flush=True)
        print("✅ System armed - Say 'guard mode' to activate", flush=True)
        print("="*70, flush=True)
        
        self.audio_thread = AudioThread(self.whisper_model, self)
        self.audio_thread.start()
        self.audio_thread.start_listening()
    
    def handle_m1(self):
        """M1: Audio thread handles everything"""
        time.sleep(0.1)
    
    def handle_m2(self):
        """M2: Face monitoring"""
        if not self.camera_thread or not self.camera_thread.running:
            return
        
        with self.camera_thread.lock:
            results = self.camera_thread.faces_data.copy()
        
        if not results:
            self.unknown_timer.update(face_present=False)
            time.sleep(0.1)
            return
        
        has_trusted = any(r['name'] != "Unknown" for r in results)
        has_unknown = any(r['name'] == "Unknown" for r in results)
        
        if has_trusted and has_unknown:
            self.unknown_timer.update(face_present=False)
            return
        
        if has_trusted and not has_unknown:
            self.unknown_timer.update(face_present=False)
            
            transcriptions = self.audio_thread.get_deactivation_commands()
            
            for trans in transcriptions:
                similarity, matched_phrase = calculate_similarity(trans, DEACTIVATION_PHRASES)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    print(f"   💬 Deactivation heard: '{trans}' (Similarity: {similarity:.2f})", flush=True)
                    print("\n✅ Deactivation (trusted face present)", flush=True)
                    
                    # ✅ FIX 2: Pause audio before TTS
                    self.audio_thread.stop_listening()
                    time.sleep(0.3)
                    
                    self.speak("System deactivated")
                    
                    self.camera_thread.stop()
                    self.face_thread.stop()
                    self.camera_thread.join(timeout=2)
                    self.face_thread.join(timeout=2)
                    
                    self.audio_thread.clear_deactivation_buffer()
                    
                    self.state = State.IDLE
                    return
        
        if has_unknown and not has_trusted:
            if self.cooldown_active:
                if time.time() - self.cooldown_start < ESCALATION_TIMING["cooldown_duration"]:
                    return
                else:
                    self.cooldown_active = False
                    print("✅ Cooldown expired", flush=True)
            
            if self.unknown_timer.update(face_present=True):
                print(f"\n⚠️  Unknown present {ESCALATION_TIMING['level1_presence']}s - ESCALATING", flush=True)
                self.state = State.M3_ESCALATION
                return
        
        time.sleep(0.1)
    
    def handle_m3(self):
        """M3: Complete escalation (L1→L2→L3)"""
        print("\n" + "="*70, flush=True)
        print("🚨 ESCALATION PROTOCOL", flush=True)
        print("="*70, flush=True)
        
        self.conversation_context.reset()
        all_speech = []
        
        self.audio_thread.stop_listening()
        
        try:
            # LEVEL 1
            print("\n" + "="*40, flush=True)
            print("📍 LEVEL 1: Initial Contact", flush=True)
            print("="*40, flush=True)
            
            response1 = self.generate_llm_response(1, "", self.conversation_context)
            self.speak(response1)
            
            speech1 = self.listen_with_silence_detection(countdown=5, max_duration=10)
            if speech1:
                all_speech.append(speech1)
                self.conversation_context.update_from_speech(speech1)
            
            with self.camera_thread.lock:
                results = self.camera_thread.faces_data.copy()
            if any(r['name'] != "Unknown" for r in results):
                print("\n✅ Trusted entered - Dropping escalation", flush=True)
                self.state = State.M2_FACE
                self.unknown_timer.reset()
                self.audio_thread.clear_deactivation_buffer()
                self.audio_thread.start_listening()
                return
            
            # LEVEL 2
            print("\n⏳ Waiting for 4s accumulated presence...", flush=True)
            
            while self.unknown_timer.accumulated < ESCALATION_TIMING["level2_presence"]:
                with self.camera_thread.lock:
                    results = self.camera_thread.faces_data.copy()
                
                has_trusted = any(r['name'] != "Unknown" for r in results)
                has_unknown = any(r['name'] == "Unknown" for r in results)
                
                if has_trusted:
                    print("\n✅ Trusted entered - Dropping escalation", flush=True)
                    self.state = State.M2_FACE
                    self.unknown_timer.reset()
                    self.audio_thread.clear_deactivation_buffer()
                    self.audio_thread.start_listening()
                    return
                
                if not has_unknown:
                    self.unknown_timer.update(face_present=False)
                    if not self.unknown_timer.is_active:
                        print("\n⚠️  Unknown left >7s - Dropping escalation", flush=True)
                        self.state = State.M2_FACE
                        self.audio_thread.clear_deactivation_buffer()
                        self.audio_thread.start_listening()
                        return
                else:
                    self.unknown_timer.update(face_present=True)
                
                time.sleep(0.1)
            
            print(f"✅ Timer: {self.unknown_timer.accumulated:.1f}s (Level 2 triggered)", flush=True)
            
            print("\n" + "="*40, flush=True)
            print("📍 LEVEL 2: Context-Aware Warning", flush=True)
            print("="*40, flush=True)
            
            prev = all_speech[-1] if all_speech else ""
            response2 = self.generate_llm_response(2, prev, self.conversation_context)
            self.speak(response2)
            
            speech2 = self.listen_with_silence_detection(countdown=5, max_duration=10)
            if speech2:
                all_speech.append(speech2)
                self.conversation_context.update_from_speech(speech2)
            
            with self.camera_thread.lock:
                results = self.camera_thread.faces_data.copy()
            if any(r['name'] != "Unknown" for r in results):
                print("\n✅ Trusted entered - Dropping escalation", flush=True)
                self.state = State.M2_FACE
                self.unknown_timer.reset()
                self.audio_thread.clear_deactivation_buffer()
                self.audio_thread.start_listening()
                return
            
            # LEVEL 3
            print("\n⏳ Waiting for 6s accumulated presence...", flush=True)
            
            while self.unknown_timer.accumulated < ESCALATION_TIMING["level3_presence"]:
                with self.camera_thread.lock:
                    results = self.camera_thread.faces_data.copy()
                
                has_trusted = any(r['name'] != "Unknown" for r in results)
                has_unknown = any(r['name'] == "Unknown" for r in results)
                
                if has_trusted:
                    print("\n✅ Trusted entered - Dropping escalation", flush=True)
                    self.state = State.M2_FACE
                    self.unknown_timer.reset()
                    self.audio_thread.clear_deactivation_buffer()
                    self.audio_thread.start_listening()
                    return
                
                if not has_unknown:
                    self.unknown_timer.update(face_present=False)
                    if not self.unknown_timer.is_active:
                        print("\n⚠️  Unknown left >7s - Dropping escalation", flush=True)
                        self.state = State.M2_FACE
                        self.audio_thread.clear_deactivation_buffer()
                        self.audio_thread.start_listening()
                        return
                else:
                    self.unknown_timer.update(face_present=True)
                
                time.sleep(0.1)
            
            print(f"✅ Timer: {self.unknown_timer.accumulated:.1f}s (Level 3 triggered)", flush=True)
            
            print("\n" + "="*40, flush=True)
            print("📍 LEVEL 3: Final Warning", flush=True)
            print("="*40, flush=True)
            
            prev = all_speech[-1] if all_speech else ""
            response3 = self.generate_llm_response(3, prev, self.conversation_context)
            self.speak(response3)
            
            print("\n✅ Protocol complete - 30s cooldown activated", flush=True)
            
            self.cooldown_active = True
            self.cooldown_start = time.time()
            self.state = State.M2_FACE
            self.unknown_timer.reset()
            self.audio_thread.clear_deactivation_buffer()
            self.audio_thread.start_listening()
        
        except Exception as e:
            print(f"\n❌ Escalation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.state = State.M2_FACE
            self.unknown_timer.reset()
            self.audio_thread.clear_deactivation_buffer()
            self.audio_thread.start_listening()
    
    def run(self):
        """Main loop"""
        try:
            while not self.shutdown_requested:
                if self.state == State.IDLE:
                    self.handle_idle()
                
                elif self.state == State.M1_VOICE:
                    self.handle_m1()
                
                elif self.state == State.M2_FACE:
                    self.handle_m2()
                
                elif self.state == State.M3_ESCALATION:
                    self.handle_m3()
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Keyboard interrupt", flush=True)
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🧹 Cleaning up...", flush=True)
        
        if self.camera_thread:
            self.camera_thread.stop()
            cv2.destroyAllWindows()
            self.camera_thread.join(timeout=1)
        
        if self.face_thread:
            self.face_thread.stop()
            self.face_thread.join(timeout=1)
        
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.join(timeout=1)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Cleanup complete", flush=True)
        sys.exit(0)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        bot = GuardBotStateMachine()
        bot.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
