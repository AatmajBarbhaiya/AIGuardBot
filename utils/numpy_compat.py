"""
NumPy 2.0 compatibility layer
"""
import numpy as np
import warnings

class NumPyCompatibility:
    """Handle NumPy 2.0 compatibility issues"""
    
    @staticmethod
    def safe_array_creation(array_data, dtype=None):
        """Safely create numpy arrays compatible with NumPy 2.0"""
        try:
            if hasattr(array_data, '__array__'):
                array = np.asarray(array_data, dtype=dtype)
            else:
                array = np.array(array_data, dtype=dtype)
            return array
        except (ValueError, TypeError) as e:
            warnings.warn(f"Array creation warning: {e}, attempting fallback")
            # Fallback: ensure we have a proper numeric array
            if dtype is None:
                dtype = np.float32
            return np.array(array_data, dtype=dtype, copy=True)
    
    @staticmethod
    def torch_to_numpy_safe(tensor):
        """Safely convert torch tensor to numpy for NumPy 2.0"""
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach()
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if hasattr(tensor, 'numpy'):
            try:
                return tensor.numpy()
            except Exception as e:
                warnings.warn(f"Direct tensor conversion failed: {e}, using manual conversion")
                return NumPyCompatibility.safe_array_creation(tensor)
        else:
            return NumPyCompatibility.safe_array_creation(tensor)

def ensure_numpy_compatible(array_like):
    """
    Convert array-like objects to NumPy 2.0 compatible format
    """
    return NumPyCompatibility.safe_array_creation(array_like)

def torch_to_numpy(tensor):
    """Safely convert torch tensor to numpy array for NumPy 2.0"""
    return NumPyCompatibility.torch_to_numpy_safe(tensor)