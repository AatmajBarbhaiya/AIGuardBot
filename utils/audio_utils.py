"""
Audio utility functions with PyAudio (more reliable on Windows)
Updated for NumPy 2.0 compatibility - FIXED VAD with proper 2s silence detection
"""

import numpy as np
import pyaudio
import logging
from collections import deque
from dataclasses import dataclass
import time
import os
import sys

# Fix import path for config
try:
    # When running from milestone1 (outside utils folder)
    from utils.config import (
        SAMPLE_RATE, CHANNELS, FRAME_MS, PRE_ROLL_MS, POST_ROLL_MS,
        ENERGY_SMOOTHING, SILENCE_THRESHOLD, MIN_SPEECH_MS
    )
except ImportError:
    # When running from within utils folder
    from config import (
        SAMPLE_RATE, CHANNELS, FRAME_MS, PRE_ROLL_MS, POST_ROLL_MS,
        ENERGY_SMOOTHING, SILENCE_THRESHOLD, MIN_SPEECH_MS
    )

logger = logging.getLogger(__name__)

@dataclass
class VADParams:
    frame_ms: int = FRAME_MS
    pre_roll_ms: int = PRE_ROLL_MS
    post_roll_ms: int = POST_ROLL_MS
    energy_smoothing: float = ENERGY_SMOOTHING
    silence_threshold: float = SILENCE_THRESHOLD
    min_speech_ms: int = MIN_SPEECH_MS


class StreamVAD:
    """
    Streaming voice activity detector using PyAudio
    NumPy 2.0 compatible version - FIXED (no device index parameter)
    """

    def __init__(self, params: VADParams = VADParams()):
        self.params = params
        self.frame_len = int(self.params.frame_ms * SAMPLE_RATE / 1000)
        self.pre_roll_frames = int(self.params.pre_roll_ms * SAMPLE_RATE / 1000)
        self.post_roll_frames = int(self.params.post_roll_ms * SAMPLE_RATE / 1000)
        self.min_speech_frames = int(self.params.min_speech_ms * SAMPLE_RATE / 1000)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.buffer_pre = deque(maxlen=self.pre_roll_frames)
        self.buffer_out = []
        self.energy = 0.0
        self.speaking = False
        self.silence_hold = 0
        self.is_running = False

    def _update_energy(self, frame_i16: np.ndarray):
        # Ensure proper array format for NumPy 2.0
        f = np.asarray(frame_i16, dtype=np.float32).flatten()
        rms = np.sqrt(np.mean(f * f)) / 32768.0
        self.energy = self.params.energy_smoothing * self.energy + (1 - self.params.energy_smoothing) * rms
        return self.energy

    def stream_chunks(self):
        """
        Generator: yields numpy int16 arrays per detected utterance
        """
        self.is_running = True
        # Use default device without specifying index
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.frame_len,
            stream_callback=None
        )

        self.stream.start_stream()

        try:
            while self.is_running:
                try:
                    data = self.stream.read(self.frame_len, exception_on_overflow=False)
                    # Use frombuffer with explicit dtype for NumPy 2.0 compatibility
                    frame = np.frombuffer(data, dtype=np.int16)

                    # Keep pre-roll
                    self.buffer_pre.extend(frame)

                    e = self._update_energy(frame)

                    if not self.speaking:
                        if e >= self.params.silence_threshold:
                            # Start of speech
                            self.speaking = True
                            self.buffer_out = list(self.buffer_pre)  # Include pre-roll
                            self.silence_hold = 0
                            logger.debug("Speech started")
                    else:
                        # Within speech
                        self.buffer_out.extend(frame)

                        if e < self.params.silence_threshold:
                            self.silence_hold += len(frame)

                            # If we've been silent long enough, close the chunk
                            if self.silence_hold >= self.post_roll_frames:
                                # Ensure proper array creation for NumPy 2.0
                                out = np.array(self.buffer_out, dtype=np.int16)
                                if len(out) >= self.min_speech_frames:
                                    logger.debug(f"Speech chunk: {len(out)} frames")
                                    yield out

                                # Reset
                                self.speaking = False
                                self.buffer_out = []
                                self.silence_hold = 0
                        else:
                            self.silence_hold = 0

                except Exception as e:
                    if self.is_running:  # Only log if we're supposed to be running
                        logger.warning(f"Stream read error: {e}")
                    break

        except Exception as e:
            logger.error(f"VAD error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the VAD stream"""
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        try:
            self.audio.terminate()
        except:
            pass


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 array to float32, NumPy 2.0 compatible"""
    # Ensure proper array conversion
    x_array = np.asarray(x, dtype=np.int16)
    return (x_array.astype(np.float32).flatten()) / 32768.0
