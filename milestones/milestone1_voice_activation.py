import os
import sys
import time
import queue
import threading
import torch
import whisper
import pyttsx3
from difflib import SequenceMatcher

# Fix import paths - add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from utils
try:
    from utils.numpy_compat import NumPyCompatibility
    from utils.audio_utils import StreamVAD, VADParams, int16_to_float32
    from utils.video_utils import CameraManager
    from utils.state_manager import StateManager
    from utils.config import AgentState
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all files are in the correct locations:")
    print("- numpy_compat.py in utils/ folder")
    print("- audio_utils.py in utils/ folder") 
    print("- video_utils.py in utils/ folder")
    print("- state_manager.py in utils/ folder")
    print("- config.py in utils/ folder")
    sys.exit(1)

class ContinuousVoiceActivation:
    def __init__(self, model_size="medium"):
        print("🚀 Initializing Continuous Voice Activation System...")
        
        # Check and setup GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ Using CPU - GPU not available")
        
        # Load Whisper model on GPU
        print("📥 Loading Whisper medium model (1.4GB)...")
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Whisper loading failed: {e}")
            sys.exit(1)
        
        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Voice Activity Detection - TIGHTER SETTINGS
        self.vad_params = VADParams(
            frame_ms=30,
            pre_roll_ms=200,  # Reduced from 300
            post_roll_ms=300,  # Reduced from 400
            silence_threshold=0.003,  # Increased threshold - less sensitive
            min_speech_ms=800  # Increased - require longer speech
        )
        self.vad = StreamVAD(self.vad_params)
        
        # Audio chunk queue for continuous processing
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_listening = False
        self.producer_thread = None
        
        # State management
        self.state_manager = StateManager()
        self.camera_manager = CameraManager()
        
        # Command recognition
        self.activation_phrases = [
            "guard my room", "start guarding", "activate guard", 
            "guard mode", "secure my room", "begin surveillance"
        ]
        self.deactivation_phrases = [
            "stand down", "stop guarding", "deactivate guard", 
            "guard off", "stop security", "cease monitoring"
        ]
        self.similarity_threshold = 0.6
        
        # Command cooldown to prevent spam
        self.last_command_time = 0
        self.command_cooldown = 3.0  # seconds
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"🤖 AI: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def calculate_similarity(self, text, phrases):
        """Calculate similarity between transcribed text and command phrases"""
        if not text or not text.strip():
            return 0.0, ""
        
        text = text.lower().strip()
        best_similarity = 0.0
        best_phrase = ""
        
        for phrase in phrases:
            # Sequence similarity
            seq_similarity = SequenceMatcher(None, text, phrase).ratio()
            
            # Word overlap
            text_words = set(text.split())
            phrase_words = set(phrase.split())
            word_similarity = len(text_words.intersection(phrase_words)) / max(len(phrase_words), 1)
            
            # Combined score (weighted towards sequence similarity)
            combined_score = 0.7 * seq_similarity + 0.3 * word_similarity
            
            if combined_score > best_similarity:
                best_similarity = combined_score
                best_phrase = phrase
                
        return best_similarity, best_phrase
    
    def audio_producer(self):
        """Background thread that continuously produces audio chunks from VAD"""
        print("🎤 Audio producer started - continuously listening...")
        try:
            for chunk in self.vad.stream_chunks():
                if not self.is_listening:
                    break
                try:
                    # Put chunk in queue for processing
                    self.audio_queue.put(chunk, timeout=0.1)
                except queue.Full:
                    # Drop chunk if queue is full (system busy)
                    pass
        except Exception as e:
            print(f"❌ Audio producer error: {e}")
    
    def transcribe_audio(self, audio_chunk):
        """Transcribe audio chunk using Whisper on GPU"""
        try:
            # Convert to float32 for Whisper
            audio_float = int16_to_float32(audio_chunk)
            
            # Use GPU with fp16 for faster inference
            fp16 = self.device == "cuda"
            
            # Transcribe with timing
            start_time = time.time()
            result = self.model.transcribe(
                audio_float, 
                fp16=fp16, 
                language='en',
                no_speech_threshold=0.7,  # Increased - ignore more non-speech
                logprob_threshold=-0.5    # Added - require higher confidence
            )
            latency = (time.time() - start_time) * 1000
            
            text = result.get("text", "").strip()
            
            # Filter out very short or likely false transcripts
            if len(text) < 3:  # Ignore very short texts
                return "", latency
                
            return text, latency
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return "", 0
    
    def start_continuous_listening(self):
        """Start continuous voice activation system"""
        print("\n" + "="*70)
        print("🚀 CONTINUOUS VOICE ACTIVATION SYSTEM READY")
        print(f"🎯 Activation: {', '.join(self.activation_phrases[:3])}...")
        print(f"🎯 Deactivation: {', '.join(self.deactivation_phrases[:3])}...")
        print("🔊 Listening continuously with VAD (Voice Activity Detection)")
        print("💻 Using GPU for real-time Whisper transcription")
        print("⏰ Command cooldown: 3 seconds to prevent spam")
        print("="*70 + "\n")
        
        self.is_listening = True
        self.state_manager.set_state(AgentState.LISTENING, "System startup")
        
        # Start audio producer thread
        self.producer_thread = threading.Thread(target=self.audio_producer, daemon=True)
        self.producer_thread.start()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received...")
        finally:
            self.stop_listening()
    
    def _main_loop(self):
        """Main processing loop"""
        while self.is_listening:
            current_state = self.state_manager.get_state()
            
            # Process audio chunks if available
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Check cooldown to avoid processing too frequently
                current_time = time.time()
                if current_time - self.last_command_time < self.command_cooldown:
                    continue
                
                # Transcribe audio
                transcription, latency = self.transcribe_audio(audio_chunk)
                
                if transcription:
                    print(f"💬 Heard: '{transcription}' (Latency: {latency:.0f}ms)")
                    
                    # State-specific command processing
                    if current_state == AgentState.LISTENING:
                        self._process_activation_command(transcription)
                    elif current_state == AgentState.ACTIVATED:
                        self._process_deactivation_command(transcription)
                        
                    self.last_command_time = current_time
                
            except queue.Empty:
                # No audio to process, continue
                pass
            
            # Handle camera display if active
            if current_state == AgentState.ACTIVATED:
                self._update_camera_display()
            
            # If we're in IDLE state after deactivation, stop the system
            if current_state == AgentState.IDLE:
                print("🛑 System returning to idle state - stopping...")
                break
    
    def _process_activation_command(self, transcription):
        """Process potential activation commands"""
        similarity, matched_phrase = self.calculate_similarity(transcription, self.activation_phrases)
        
        if similarity >= self.similarity_threshold:
            print(f"🎯 ACTIVATION DETECTED! (Similarity: {similarity:.2f}, Matched: '{matched_phrase}')")
            self.speak(f"Guard mode activated! I am now monitoring this room.")
            self.state_manager.set_state(AgentState.ACTIVATED, "Voice activation command")
            
            # Start camera
            if not self.camera_manager.start():
                print("⚠️  Camera failed to start, but guard mode is active")
    
    def _process_deactivation_command(self, transcription):
        """Process potential deactivation commands"""
        similarity, matched_phrase = self.calculate_similarity(transcription, self.deactivation_phrases)
        
        if similarity >= self.similarity_threshold:
            print(f"🛑 DEACTIVATION DETECTED! (Similarity: {similarity:.2f}, Matched: '{matched_phrase}')")
            self.speak("Guard mode deactivated. Standing down.")
            self.state_manager.set_state(AgentState.IDLE, "Voice deactivation command")
            self.camera_manager.stop()
    
    def _update_camera_display(self):
        """Update camera display when guard is active"""
        frame = self.camera_manager.read_frame()
        if frame is not None:
            self.camera_manager.display_frame(frame)
    
    def stop_listening(self):
        """Stop the continuous listening system"""
        print("\n🛑 Stopping continuous listening...")
        self.is_listening = False
        
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=1.0)
        
        self.camera_manager.stop()
        self.state_manager.set_state(AgentState.IDLE, "System shutdown")
        print("✅ System stopped successfully")

def main():
    print("🤖 AI Room Guard - Milestone 1: Continuous Voice Activation (GPU)")
    
    # Initialize system
    voice_system = ContinuousVoiceActivation(model_size="medium")
    
    try:
        # Start continuous listening
        voice_system.start_continuous_listening()
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("\n🔚 System shutdown complete")

if __name__ == "__main__":
    main()