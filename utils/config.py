"""
Configuration constants for AI Guard Agent
"""
from pathlib import Path
from enum import Enum

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"
MILESTONES_DIR = PROJECT_ROOT / "milestones"
LOGS_DIR.mkdir(exist_ok=True)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# Streaming VAD params
FRAME_MS = 30
PRE_ROLL_MS = 300
POST_ROLL_MS = 400
ENERGY_SMOOTHING = 0.9
SILENCE_THRESHOLD = 0.0018
MIN_SPEECH_MS = 500

# Camera settings - OPTIMIZED FOR HIGH FPS
DEFAULT_CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60  # Increased for smoother video
CAMERA_BUFFER_SIZE = 1  # Reduced buffer for lower latency

# Agent States
class AgentState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    ACTIVATED = "activated"
    MONITORING = "monitoring"

# Logging
LOG_FILE = LOGS_DIR / "agent.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Face Recognition Settings (Using face_recognition library)
TRUSTED_FACES_DIR = DATA_DIR / "trusted_faces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "known_faces.pkl"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# face_recognition settings
FACE_RECOGNITION_MODEL = "cnn"  # "hog" (faster, CPU) or "cnn" (more accurate, GPU/CUDA)
FACE_RECOGNITION_TOLERANCE = 0.4  # Lower = more strict (0.6 is default)
FACE_DETECTION_UPSAMPLES = 1  # Number of times to upsample the image (0-2, higher = find smaller faces)

# GPU acceleration settings
USE_GPU_FACE_RECOGNITION = True  # Use GPU for face recognition if available
GPU_MEMORY_FRACTION = 0.5  # Fraction of GPU memory to use

# Face recognition performance settings
FACE_DETECTION_INTERVAL = 5  # Process every 5th frame for better performance
MIN_FACE_CONFIDENCE = 0.8  # Minimum confidence for face detection

# Create necessary directories
TRUSTED_FACES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Face recognition cache settings
ENABLE_FACE_CACHE = True
FACE_CACHE_DURATION = 300  # Cache faces for 5 minutes

# Display settings
DISPLAY_FPS = True
DISPLAY_FACE_BOXES = True
DISPLAY_RECOGNITION_INFO = True
DISPLAY_TIMESTAMP = True

# Enrollment settings
ENROLLMENT_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MIN_FACES_PER_PERSON = 1
MAX_FACES_PER_PERSON = 10

#Enhanced Face Recognition Settings
ENHANCED_RECOGNITION_SETTINGS = {
    'known_face_tolerance': 0.55,      # Slightly stricter for known faces
    'unknown_face_tolerance': 0.4,     # Much stricter for unknown faces
    'min_confidence_known': 0.70,      # Minimum confidence to show as known
    'confidence_boost_factor': 1.1,    # Boost confidence for quality embeddings
    'temporal_smoothing_frames': 5,    # Frames to smooth confidence
    'min_consistent_matches': 3,       # Required consistent matches
    'recognition_interval': 1.0,       # Recognize every 1 second
}

# Enhanced Enrollment Settings
ENROLLMENT_SETTINGS = {
    'min_images_per_person': 8,
    'max_images_per_person': 15,
    'optimal_images_per_person': 12,
    'min_face_quality_score': 0.7,
    'encoding_model': 'large',  # More accurate encodings
    'required_angles': ['center', 'left', 'right', 'slight_up', 'slight_down'],
    'lighting_conditions': ['normal', 'bright']
}