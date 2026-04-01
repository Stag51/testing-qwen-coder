"""Utility functions for the Federated Diagnostic Swarm"""
import hashlib
from typing import Any, Dict, List
import json
from datetime import datetime
from loguru import logger


def generate_patient_hash(patient_id: str, salt: str = "federated_swarm") -> str:
    """
    Generate a privacy-preserving hash for patient ID
    
    Args:
        patient_id: Original patient identifier
        salt: Salt for hashing
        
    Returns:
        SHA-256 hash of patient ID
    """
    combined = f"{patient_id}:{salt}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def serialize_tensor_info(tensor) -> Dict[str, Any]:
    """
    Serialize tensor information for API responses
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Dictionary with tensor metadata
    """
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad
    }


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime as ISO string"""
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + "Z"


def validate_fhir_resource(resource: Dict) -> bool:
    """
    Basic validation for FHIR resources
    
    Args:
        resource: FHIR resource dictionary
        
    Returns:
        True if valid
    """
    required_fields = ["resourceType", "id"]
    return all(field in resource for field in required_fields)


def calculate_model_size(state_dict: Dict) -> int:
    """
    Calculate total model size in parameters
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in state_dict.values()) if state_dict else 0


class PrivacyLogger:
    """
    Logger that ensures no PHI (Protected Health Information) is logged
    """
    
    def __init__(self):
        self.logger = logger
    
    def info(self, message: str, **kwargs):
        """Log info message after sanitizing"""
        sanitized = self._sanitize(message)
        self.logger.info(sanitized, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message after sanitizing"""
        sanitized = self._sanitize(message)
        self.logger.warning(sanitized, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message after sanitizing"""
        sanitized = self._sanitize(message)
        self.logger.error(sanitized, **kwargs)
    
    def _sanitize(self, text: str) -> str:
        """Remove potential PHI from text"""
        # Simple sanitization - in production, use more robust methods
        phi_patterns = [
            "patient_id",
            "PatientID",
            "name",
            "address",
            "phone",
            "email",
            "ssn",
            "mrn"
        ]
        
        sanitized = text
        for pattern in phi_patterns:
            if pattern.lower() in sanitized.lower():
                sanitized = sanitized.replace(pattern, f"[{pattern.upper()}_REDACTED]")
        
        return sanitized


# Global privacy-safe logger
privacy_logger = PrivacyLogger()
