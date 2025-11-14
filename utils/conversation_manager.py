"""
Enhanced Conversation Manager - SIMPLIFIED VERSION
Simple first-person responses and basic tracking
"""
import time
import logging
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import cv2
import os

logger = logging.getLogger(__name__)

class IntruderTracker:
    """Simple intruder tracking"""
    
    def __init__(self):
        self.intruder_speech_log: List[Dict] = []
        self.current_level = 0
        
    def log_intruder_speech(self, speech_text: str) -> None:
        """Log intruder speech"""
        if speech_text and speech_text.strip():
            entry = {
                "timestamp": datetime.now().isoformat(),
                "speech": speech_text.strip(),
                "level": self.current_level
            }
            self.intruder_speech_log.append(entry)
            logger.info(f"🎤 Intruder speech: '{speech_text.strip()}'")
    
    def get_recent_speech(self, last_n: int = 2) -> List[str]:
        """Get recent intruder speech"""
        return [entry['speech'] for entry in self.intruder_speech_log[-last_n:]]
    
    def reset(self):
        """Reset tracker"""
        self.intruder_speech_log = []
        self.current_level = 0


class ConversationManager:
    """Manages simple conversation with first-person responses"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.is_loaded = False
        self.intruder_tracker = IntruderTracker()
        self.personality_system = None
        
        # Simple fallback responses
        self.FALLBACK_RESPONSES = {
            "trusted": "Welcome back. I'm watching the room.",
            "unknown_level1": "I don't know you. Who are you and why are you here?",
            "unknown_level2": "You need to leave right now.",
            "unknown_level3": "I'm calling security. Leave immediately."
        }
    
    def load_llm(self) -> bool:
        """Load Ollama LLM"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_loaded = True
                logger.info("✅ Ollama LLM loaded")
                return True
            else:
                logger.error("❌ Ollama not running")
                return False
        except:
            logger.error("❌ Ollama connection failed")
            return False
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama"""
        if not self.is_loaded:
            return ""
            
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 30}  # Shorter responses
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            return ""
        except:
            return ""
    
    def log_intruder_speech(self, speech_text: str):
        """Log intruder speech"""
        self.intruder_tracker.log_intruder_speech(speech_text)
    
    def reset_session(self):
        """Reset session data"""
        self.intruder_tracker.reset()