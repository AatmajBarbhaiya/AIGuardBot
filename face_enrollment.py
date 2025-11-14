"""
Enhanced Face Enrollment Script for GPU Backend (FaceNet)
High-quality enrollment with FaceNet 512D embeddings on GPU
Generates robust embeddings compatible with GPU face recognition
"""

import os
import pickle
import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from pathlib import Path
import logging
import sys
import time
import numpy as np
from datetime import datetime
import threading
import msvcrt

# Fix import path - config is in utils folder
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "utils"))

try:
    from utils.config import (
        TRUSTED_FACES_DIR, EMBEDDINGS_FILE,
        ENROLLMENT_IMAGE_EXTENSIONS, CAMERA_WIDTH, CAMERA_HEIGHT
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure config.py is in the utils/ folder")
    sys.exit(1)

# Enhanced enrollment settings for GPU
ENROLLMENT_SETTINGS = {
    'min_images_per_person': 8,
    'max_images_per_person': 15,
    'optimal_images_per_person': 12,
    'min_face_quality_score': 0.5,
    'encoding_model': 'facenet',  # Using FaceNet GPU backend
    'required_angles': ['center', 'left', 'right', 'slight_up', 'slight_down'],
    'lighting_conditions': ['normal', 'bright']
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KeyboardThread(threading.Thread):
    """Thread to handle keyboard input independently of OpenCV"""

    def __init__(self):
        super().__init__()
        self.daemon = True
        self.last_key = None
        self.running = True
        self.start()

    def run(self):
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                self.last_key = key
            time.sleep(0.05)

    def get_key(self):
        key = self.last_key
        self.last_key = None
        return key

    def stop(self):
        self.running = False


class EnhancedFaceEnrollmentGPU:
    """GPU-accelerated face enrollment using FaceNet"""

    def __init__(self, device=None):
        """Initialize GPU enrollment system"""
        print("\n" + "="*70)
        print("🎭 GPU FACE ENROLLMENT SYSTEM - INITIALIZATION")
        print("="*70)

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"✅ Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

        self.known_face_encodings = []
        self.known_face_names = []
        self.face_quality_metrics = []
        self.camera = None
        self.person_name = ""
        self.person_folder = None
        self.image_counter = 0
        self.captured_images = []
        self.keyboard_thread = None

        # Initialize GPU models
        print("\n📥 Loading GPU Models...")
        try:
            # MTCNN for face detection
            print("  • Loading MTCNN detector...")
            self.detector = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device,
                keep_all=True
            )
            print("    ✅ MTCNN detector loaded")

            # InceptionResnetV1 for encoding
            print("  • Loading InceptionResnetV1 encoder...")
            self.encoder = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
            print("    ✅ FaceNet encoder loaded (512D embeddings)")

        except Exception as e:
            logger.error(f"❌ Failed to load GPU models: {e}")
            sys.exit(1)

        print(f"✅ GPU models loaded successfully!")

        self.load_existing_embeddings()

    def load_existing_embeddings(self):
        """Load existing face embeddings with quality metrics"""
        if EMBEDDINGS_FILE.exists():
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    data = pickle.load(f)

                self.known_face_encodings = data.get('encodings', [])
                self.known_face_names = data.get('names', [])
                self.face_quality_metrics = data.get('quality_metrics', [])

                logger.info(f"✅ Loaded {len(self.known_face_names)} existing face embeddings with quality metrics")
                for metric in self.face_quality_metrics:
                    person = metric.get('person', 'Unknown')
                    quality = metric.get('average_quality', 0)
                    encodings = metric.get('num_encodings', 0)
                    logger.info(f"   👤 {person}: Quality {quality:.2f}, {encodings} encodings")
            except Exception as e:
                logger.error(f"❌ Error loading embeddings: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
                self.face_quality_metrics = []
        else:
            logger.info("ℹ️ No existing embeddings found. Starting fresh.")

    def save_embeddings(self):
        """Save face embeddings with quality metrics and version tag"""
        try:
            EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names,
                    'quality_metrics': self.face_quality_metrics,
                    'timestamp': datetime.now().isoformat(),
                    'version': 'facenet_gpu_v1',  # GPU FaceNet version
                    'embedding_dim': 512  # FaceNet produces 512D embeddings
                }, f)
            logger.info(f"💾 Saved {len(self.known_face_names)} enhanced face embeddings (FaceNet GPU)")
        except Exception as e:
            logger.error(f"❌ Error saving embeddings: {e}")

    def calculate_face_quality_score(self, face_image, face_location):
        """Calculate quality score for face detection"""
        try:
            x1, y1, x2, y2 = face_location
            face_height = y2 - y1
            face_width = x2 - x1

            # Score based on face size (larger is better)
            # Lower threshold: 100x100 pixels minimum (vs 200x200)
            size_score = min(face_height * face_width / (100 * 100), 1.0)

            # Score based on aspect ratio (close to 1:1 is better)
            # Relaxed tolerance for aspect ratio
            aspect_ratio = face_width / face_height
            aspect_score = max(0.5, 1.0 - min(abs(aspect_ratio - 1.0), 0.5))

            # Score based on image sharpness
            # More lenient sharpness threshold
            gray_face = cv2.cvtColor(face_image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 500.0, 1.0)  # Lowered from 1000

            # Overall quality score (with better weighting)
            quality_score = (size_score * 0.35 + aspect_score * 0.25 + sharpness_score * 0.40)

            return quality_score
        except Exception as e:
            logger.debug(f"Quality calculation error: {e}")
            return 0.5


    def create_super_embedding(self, encodings):
        """Create a super embedding by averaging high-quality 512D FaceNet encodings"""
        if not encodings:
            return None

        # Stack all encodings
        encoding_stack = torch.stack(encodings)  # [N, 512]

        # Average across all encodings
        super_embedding = torch.mean(encoding_stack, dim=0)  # [512]

        # L2 normalize
        super_embedding = F.normalize(super_embedding, p=2, dim=0)

        return super_embedding.detach().cpu().numpy()

    def setup_person_folder(self, person_name):
        """Create folder for the person"""
        self.person_name = person_name.strip()
        self.person_folder = TRUSTED_FACES_DIR / self.person_name
        self.person_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created folder for {self.person_name}: {self.person_folder}")
        return self.person_folder

    def initialize_camera(self):
        """Initialize camera for high-quality capture"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("❌ Cannot open camera")
            return False

        # Set camera properties for high quality
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
        self.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)

        # Allow camera to warm up
        logger.info("📹 Camera initialized - warming up...")
        for i in range(30):
            ret, frame = self.camera.read()
            if not ret:
                continue
            time.sleep(0.05)

        # Start keyboard thread
        self.keyboard_thread = KeyboardThread()
        time.sleep(0.5)

        logger.info("📹 Camera ready for high-quality GPU enrollment")
        return True

    def get_enrollment_guidance(self, current_count):
        """Provide guidance for optimal enrollment"""
        guidance = [
            "Look straight at camera",
            "Turn head slightly LEFT",
            "Turn head slightly RIGHT",
            "Chin UP slightly",
            "Chin DOWN slightly",
            "Move to BRIGHT lighting",
            "Move to NORMAL lighting",
            "Neutral expression",
            "Slight smile",
            "Different distance from camera"
        ]

        if current_count < len(guidance):
            return guidance[current_count]
        else:
            return f"Additional variation {current_count + 1}"

    def capture_high_quality_image(self, frame):
        """Capture and process high-quality image with GPU quality checks"""
        try:
            # Convert to PIL for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # GPU-accelerated face detection
            with torch.no_grad():
                boxes, _ = self.detector.detect(pil_image)

            if boxes is None or len(boxes) == 0:
                logger.warning("⚠️ No face detected in frame")
                return False

            # Use the largest face detected
            box_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            largest_idx = np.argmax(box_sizes)
            face_location = boxes[largest_idx]

            # Calculate face quality score
            quality_score = self.calculate_face_quality_score(frame, face_location)

            if quality_score < ENROLLMENT_SETTINGS['min_face_quality_score']:
                logger.warning(f"⚠️ Low quality face detected (score: {quality_score:.2f})")
                return False

            # Save the high-quality image
            self.image_counter += 1
            timestamp = int(time.time())
            filename = f"{self.person_name}_{timestamp}_{self.image_counter:03d}.jpg"
            filepath = self.person_folder / filename

            success = cv2.imwrite(str(filepath), frame)
            if not success:
                logger.error(f"❌ Failed to save image: {filename}")
                return False

            logger.info(f"📸 Saved high-quality image: {filename} (quality: {quality_score:.2f})")

            # Process for GPU encoding
            encoding_data = self.process_image_for_gpu_encoding(filepath, quality_score)

            if encoding_data:
                self.captured_images.append(encoding_data)
                return True

            return False

        except Exception as e:
            logger.warning(f"⚠️ Capture error: {e}")
            return False

    def process_image_for_gpu_encoding(self, image_path, quality_score):
        """Process image to create GPU-accelerated FaceNet encodings"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # GPU-accelerated face detection
            with torch.no_grad():
                boxes, _ = self.detector.detect(pil_image)

            if boxes is None or len(boxes) == 0:
                return None

            # Get GPU-accelerated FaceNet encodings
            encodings = []
            with torch.no_grad():
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)

                    # Extract face region
                    face_image = rgb_image[y1:y2, x1:x2]

                    # Resize to 160x160 (FaceNet requirement)
                    face_image_resized = cv2.resize(face_image, (160, 160))

                    # Convert to tensor and normalize
                    face_tensor = torch.tensor(face_image_resized, dtype=torch.float32)
                    face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
                    face_tensor = face_tensor / 255.0
                    face_tensor = (face_tensor - 0.5) / 0.5
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)

                    # Get 512D FaceNet embedding
                    embedding = self.encoder(face_tensor).squeeze(0)

                    # L2 normalize
                    embedding = F.normalize(embedding, p=2, dim=0)

                    encodings.append(embedding)

            if encodings:
                return {
                    'encodings': encodings,
                    'quality_score': quality_score,
                    'image_path': image_path,
                    'timestamp': datetime.now().isoformat()
                }

            return None

        except Exception as e:
            logger.warning(f"⚠️ Encoding error: {e}")
            return None

    def finalize_enrollment(self):
        """Finalize enrollment by creating super embeddings"""
        if not self.captured_images:
            logger.warning("⚠️ No images captured for enrollment")
            return False

        all_encodings = []
        total_quality = 0

        for img_data in self.captured_images:
            for encoding in img_data['encodings']:
                all_encodings.append(encoding)
            total_quality += img_data['quality_score']

        if not all_encodings:
            logger.error("❌ No valid encodings created")
            return False

        # Create super embedding
        super_embedding = self.create_super_embedding(all_encodings)

        if super_embedding is None:
            logger.error("❌ Failed to create super embedding")
            return False

        # Calculate average quality
        avg_quality = total_quality / len(all_encodings)

        # Add to known faces
        self.known_face_encodings.append(super_embedding)
        self.known_face_names.append(self.person_name)
        self.face_quality_metrics.append({
            'person': self.person_name,
            'average_quality': avg_quality,
            'num_encodings': len(all_encodings),
            'num_images': len(self.captured_images),
            'timestamp': datetime.now().isoformat(),
            'embedding_dim': 512  # FaceNet 512D
        })

        # Save embeddings
        self.save_embeddings()

        logger.info(f"🎉 GPU-accelerated enrollment completed for {self.person_name}")
        logger.info(f"   • Images captured: {len(self.captured_images)}")
        logger.info(f"   • Total encodings: {len(all_encodings)}")
        logger.info(f"   • Average quality: {avg_quality:.2f}")
        logger.info(f"   • Embedding type: FaceNet 512D (GPU)")

        return True

    def run_gpu_enrollment(self):
        """Main GPU-accelerated enrollment process"""
        print(f"\n🎯 Starting GPU-ACCELERATED ENROLLMENT for: {self.person_name}")
        print("📸 Guidance: Capture 8-12 high-quality images with variations")
        print("   • Different angles and expressions")
        print("   • Varying lighting conditions")
        print("   • GPU-accelerated quality checking")
        print("\n🎮 CONTROLS:")
        print("   • Press 's' to CAPTURE image (GPU quality check)")
        print("   • Press 'q' to FINISH enrollment")
        print("   • Press 'ESC' or 'c' to CANCEL")
        print("\n⚠️ Make sure the console window is ACTIVE for keyboard input!\n")

        if not self.initialize_camera():
            return False

        try:
            enrollment_start_time = time.time()
            last_capture_time = 0
            capture_cooldown = 1.0

            while len(self.captured_images) < ENROLLMENT_SETTINGS['optimal_images_per_person']:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("❌ Failed to grab frame")
                    time.sleep(0.1)
                    continue

                # Display guidance and frame
                display_frame = frame.copy()

                guidance = self.get_enrollment_guidance(len(self.captured_images))
                cv2.putText(display_frame, f"Enrolling: {self.person_name}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Guidance: {guidance}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Progress: {len(self.captured_images)}/{ENROLLMENT_SETTINGS['optimal_images_per_person']}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, "Press 's' to CAPTURE (GPU quality check)",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'q' to FINISH enrollment",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show face detection with GPU acceleration
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)

                    with torch.no_grad():
                        boxes, _ = self.detector.detect(pil_image)

                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            quality_score = self.calculate_face_quality_score(frame, (x1, y1, x2, y2))
                            cv2.putText(display_frame, f"Quality: {quality_score:.2f}",
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    logger.debug(f"Face display error: {e}")

                cv2.imshow('🎭 GPU Face Enrollment - Follow Guidance (FaceNet)', display_frame)

                cv2_key = cv2.waitKey(1) & 0xFF
                thread_key = self.keyboard_thread.get_key()
                current_time = time.time()

                # Check for 's' key (capture)
                if (cv2_key == ord('s') or thread_key == 's') and (current_time - last_capture_time) > capture_cooldown:
                    last_capture_time = current_time
                    if self.capture_high_quality_image(frame):
                        success_frame = display_frame.copy()
                        cv2.putText(success_frame, "✅ CAPTURE SUCCESS!",
                                   (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(success_frame, f"Images: {len(self.captured_images)}/{ENROLLMENT_SETTINGS['optimal_images_per_person']}",
                                   (display_frame.shape[1]//2 - 120, display_frame.shape[0]//2 + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow('🎭 GPU Face Enrollment - Follow Guidance (FaceNet)', success_frame)
                        cv2.waitKey(800)

                # Check for 'q' key (finish)
                elif cv2_key == ord('q') or thread_key == 'q':
                    if len(self.captured_images) >= ENROLLMENT_SETTINGS['min_images_per_person']:
                        logger.info("🎯 Finishing enrollment as requested...")
                        break
                    else:
                        logger.warning(f"⚠️ Minimum {ENROLLMENT_SETTINGS['min_images_per_person']} images required")
                        warning_frame = display_frame.copy()
                        needed = ENROLLMENT_SETTINGS['min_images_per_person'] - len(self.captured_images)
                        cv2.putText(warning_frame, f"⚠️ Need {needed} more images!",
                                   (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow('🎭 GPU Face Enrollment - Follow Guidance (FaceNet)', warning_frame)
                        cv2.waitKey(2000)

                # Check for ESC or 'c' key (cancel)
                elif cv2_key == 27 or thread_key == 'c' or thread_key == chr(27):
                    logger.info("⏹️ Enrollment cancelled by user")
                    return False

                # Timeout check
                if time.time() - enrollment_start_time > 300:
                    logger.warning("⏰ Enrollment timeout")
                    break

            # Finalize enrollment
            if len(self.captured_images) >= ENROLLMENT_SETTINGS['min_images_per_person']:
                return self.finalize_enrollment()
            else:
                logger.error(f"❌ Insufficient images captured: {len(self.captured_images)}")
                return False

        except Exception as e:
            logger.error(f"❌ Error during enrollment: {e}")
            return False

        finally:
            if self.keyboard_thread:
                self.keyboard_thread.stop()
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()

    def list_enrolled_persons(self):
        """List all enrolled persons with quality metrics"""
        unique_persons = set(self.known_face_names)
        logger.info("📋 GPU-Enrolled Persons (FaceNet 512D):")
        for person in unique_persons:
            count = self.known_face_names.count(person)
            person_metrics = [m for m in self.face_quality_metrics if m.get('person') == person]
            if person_metrics:
                avg_quality = person_metrics[0].get('average_quality', 0)
                num_encodings = person_metrics[0].get('num_encodings', 0)
                embedding_dim = person_metrics[0].get('embedding_dim', 128)
                logger.info(f"   👤 {person}: Quality {avg_quality:.2f}, Encodings: {num_encodings}, Dim: {embedding_dim}D")
            else:
                logger.info(f"   👤 {person}: {count} embedding(s)")

    def clear_embeddings(self):
        """Clear all embeddings"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_quality_metrics = []
        if EMBEDDINGS_FILE.exists():
            EMBEDDINGS_FILE.unlink()
        logger.info("🗑️ All embeddings cleared")


def main():
    """Main enrollment function"""
    print("="*70)
    print("🎭 GPU-ACCELERATED FACE ENROLLMENT SYSTEM")
    print("   Backend: FaceNet (512D embeddings) on GPU")
    print("="*70)
    print(f"📊 Enrollment Settings:")
    print(f"   • Minimum images: {ENROLLMENT_SETTINGS['min_images_per_person']}")
    print(f"   • Optimal images: {ENROLLMENT_SETTINGS['optimal_images_per_person']}")
    print(f"   • Quality threshold: {ENROLLMENT_SETTINGS['min_face_quality_score']}")
    print(f"   • Embedding model: {ENROLLMENT_SETTINGS['encoding_model']}")
    print("="*70)

    # Initialize GPU enrollment
    try:
        enroll = EnhancedFaceEnrollmentGPU()
    except Exception as e:
        print(f"❌ Failed to initialize GPU enrollment: {e}")
        return

    while True:
        print("\nOptions:")
        print("1. Start GPU-accelerated face enrollment (Recommended)")
        print("2. List enrolled persons with quality metrics")
        print("3. Clear all embeddings")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            person_name = input("Enter the person's name to enroll: ").strip()
            if not person_name:
                print("❌ Please enter a valid name")
                continue

            enroll.setup_person_folder(person_name)
            success = enroll.run_gpu_enrollment()

            if success:
                print(f"🎉 Successfully enrolled {person_name} with GPU acceleration!")
            else:
                print(f"❌ Enrollment failed for {person_name}")

        elif choice == '2':
            enroll.list_enrolled_persons()

        elif choice == '3':
            confirm = input("Are you sure you want to clear all embeddings? (y/n): ").strip().lower()
            if confirm == 'y':
                enroll.clear_embeddings()

        elif choice == '4':
            print("👋 Exiting GPU enrollment system")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()