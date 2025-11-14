"""
Finite State Machine for Agent State Management
"""
import logging
from datetime import datetime
from threading import Lock
import os
import sys

# Fix import path for config
try:
    # When running from milestone1 (outside utils folder)
    from utils.config import AgentState, LOG_FILE, LOG_FORMAT
except ImportError:
    # When running from within utils folder
    from config import AgentState, LOG_FILE, LOG_FORMAT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class StateManager:
    """Thread-safe state manager for the AI Guard Agent"""
    
    def __init__(self):
        self.current_state = AgentState.IDLE
        self.previous_state = None
        self.state_history = []
        self.lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._log_state_change()
    
    def get_state(self):
        """Get current state"""
        with self.lock:
            return self.current_state
    
    def set_state(self, new_state, reason=""):
        """Set new state with validation and logging"""
        with self.lock:
            if new_state not in vars(AgentState).values():
                self.logger.error(f"Invalid state: {new_state}")
                return False
            
            self.previous_state = self.current_state
            self.current_state = new_state
            
            # Log state change
            self._log_state_change(reason)
            
            # Add to history
            self.state_history.append({
                'timestamp': datetime.now(),
                'from': self.previous_state,
                'to': self.current_state,
                'reason': reason
            })
            
            return True
    
    def _log_state_change(self, reason=""):
        """Log state transition"""
        if self.previous_state:
            self.logger.info(
                f"State: {self.previous_state} -> {self.current_state} | Reason: {reason}"
            )
        else:
            self.logger.info(f"Initial state: {self.current_state}")
    
    def is_state(self, state):
        """Check if current state matches"""
        with self.lock:
            return self.current_state == state
    
    def get_history(self, last_n=10):
        """Get last N state transitions"""
        with self.lock:
            return self.state_history[-last_n:]
    
    def reset(self):
        """Reset to IDLE state"""
        self.set_state(AgentState.IDLE, reason="Manual reset")