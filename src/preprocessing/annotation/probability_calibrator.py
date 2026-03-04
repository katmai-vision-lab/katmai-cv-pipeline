"""
Probability Calibration for Multi-Model Detection Consensus

Implements isotonic regression calibration to transform model confidence scores
into calibrated probabilities. This allows for more accurate combination of 
predictions from different models.

Reference: https://scikit-learn.org/stable/modules/calibration.html
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationData:
    """Stores calibration data for a single model"""
    model_name: str
    confidences: List[float] = field(default_factory=list)
    correctness: List[bool] = field(default_factory=list)  # True if detection matches ground truth
    
    def add_sample(self, confidence: float, is_correct: bool):
        """Add a single calibration sample"""
        self.confidences.append(confidence)
        self.correctness.append(is_correct)
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return numpy arrays for training"""
        return np.array(self.confidences), np.array(self.correctness, dtype=float)


class ProbabilityCalibrator:
    """
    Calibrates confidence scores from detection models using isotonic regression.
    
    For each model, learns a mapping from raw confidence → calibrated probability
    using a validation set with ground truth annotations.
    """
    
    def __init__(self):
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.is_fitted = False
    
    def fit(self, calibration_data: Dict[str, CalibrationData]):
        """
        Fit calibration curves for each model.
        
        Args:
            calibration_data: Dictionary mapping model_name → CalibrationData
        """
        print("\n=== Training Probability Calibrators ===")
        
        for model_name, data in calibration_data.items():
            if len(data.confidences) < 10:
                print(f"Warning: {model_name} has only {len(data.confidences)} samples, skipping calibration")
                continue
            
            confidences, correctness = data.get_arrays()
            
            # Isotonic regression for calibration
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(confidences, correctness)
            
            self.calibrators[model_name] = calibrator
            
            # Calculate calibration improvement
            uncalibrated_error = self._calculate_calibration_error(confidences, correctness)
            calibrated_probs = calibrator.predict(confidences)
            calibrated_error = self._calculate_calibration_error(calibrated_probs, correctness)
            
            print(f"\n{model_name}:")
            print(f"  Samples: {len(confidences)}")
            print(f"  Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
            print(f"  Uncalibrated ECE: {uncalibrated_error:.4f}")
            print(f"  Calibrated ECE: {calibrated_error:.4f}")
            print(f"  Improvement: {(uncalibrated_error - calibrated_error):.4f}")
        
        self.is_fitted = True
        print("\n✓ Calibration complete")
    
    def calibrate(self, model_name: str, confidence: float) -> float:
        """
        Apply calibration to a single confidence score.
        
        Args:
            model_name: Name of the model (gdino/megadet/detr)
            confidence: Raw confidence score [0, 1]
        
        Returns:
            Calibrated probability [0, 1]
        """
        if not self.is_fitted or model_name not in self.calibrators:
            # No calibration available, return raw confidence
            return confidence
        
        calibrated = self.calibrators[model_name].predict([confidence])[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def calibrate_batch(self, model_name: str, confidences: np.ndarray) -> np.ndarray:
        """
        Apply calibration to multiple confidence scores.
        
        Args:
            model_name: Name of the model
            confidences: Array of raw confidence scores
        
        Returns:
            Array of calibrated probabilities
        """
        if not self.is_fitted or model_name not in self.calibrators:
            return confidences
        
        calibrated = self.calibrators[model_name].predict(confidences)
        return np.clip(calibrated, 0.0, 1.0)
    
    def save(self, filepath: Path):
        """Save calibrators to disk"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'calibrators': self.calibrators,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Saved calibrators to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'ProbabilityCalibrator':
        """Load calibrators from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls()
        calibrator.calibrators = data['calibrators']
        calibrator.is_fitted = data['is_fitted']
        return calibrator
    
    def _calculate_calibration_error(
        self, 
        confidences: np.ndarray, 
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between confidence and accuracy across bins.
        Lower is better (0 = perfectly calibrated).
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correctness[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def get_calibration_stats(self) -> Dict[str, dict]:
        """Get statistics about fitted calibrators"""
        if not self.is_fitted:
            return {}
        
        stats = {}
        for model_name, calibrator in self.calibrators.items():
            stats[model_name] = {
                'n_samples': len(calibrator.X_thresholds_),
                'min_confidence': float(calibrator.X_thresholds_.min()),
                'max_confidence': float(calibrator.X_thresholds_.max()),
            }
        return stats


def analyze_calibration_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze calibration by binning confidences and computing accuracy per bin.
    
    Returns:
        bin_centers: Center of each confidence bin
        bin_accuracies: Actual accuracy in each bin
        bin_counts: Number of samples in each bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append(confidences[mask].mean())
            bin_accuracies.append(correctness[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
    
    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)
