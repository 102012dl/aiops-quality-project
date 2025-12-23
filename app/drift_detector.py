import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleDriftDetector:
    """Simple Z-score based drift detector"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.reference_mean: Optional[np.ndarray] = None
        self.reference_std: Optional[np.ndarray] = None
        self.drift_count = 0
        
    def fit(self, X: np.ndarray):
        """Fit on reference data"""
        self.reference_mean = np.mean(X, axis=0)
        self.reference_std = np.std(X, axis=0) + 1e-6
        logger.info(f"Drift detector fitted on {len(X)} samples")
        
    def detect(self, X: np.ndarray) -> Dict:
        """Detect drift using Z-score"""
        if self.reference_mean is None:
            return {"drift_detected": False, "reason": "Not fitted"}
        
        z_scores = np.abs((X - self.reference_mean) / self.reference_std)
        max_z_score = np.max(z_scores)
        
        drift_detected = max_z_score > self.threshold
        
        if drift_detected:
            self.drift_count += 1
            logger.warning(f"🚨 DRIFT DETECTED! Z-score: {max_z_score:.2f}")
        
        return {
            "drift_detected": drift_detected,
            "max_z_score": float(max_z_score),
            "threshold": self.threshold,
            "drift_count": self.drift_count
        }
