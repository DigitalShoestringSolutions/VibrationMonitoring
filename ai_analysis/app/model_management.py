# ai_analysis/app/model_management.py
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import shutil
import logging

logger = logging.getLogger(__name__)

class SimpleModelManager:
    def __init__(self):
        self.base_dir = "data/checkpoints"
        self.metadata_file = os.path.join(self.base_dir, "model_registry.json")
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize or load metadata
        if not os.path.exists(self.metadata_file):
            self._save_metadata({
                "models": [],
                "current_model": None
            })
    
    def _save_metadata(self, metadata: Dict) -> None:
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict:
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def save_model(self, 
                  model_path: str,
                  metrics: Dict[str, float],
                  params: Dict[str, Any],
                  description: str = "") -> str:
        """
        Save a model checkpoint with metadata.
        
        Args:
            model_path: Path to the model file
            metrics: Dictionary of training metrics
            params: Dictionary of training parameters
            description: Optional description of this model version
            
        Returns:
            model_id: Unique identifier for this model version
        """
        # Generate unique model ID using timestamp
        model_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Create model directory
        model_dir = os.path.join(self.base_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file
        new_model_path = os.path.join(model_dir, "model.tflite")
        shutil.copy2(model_path, new_model_path)
        
        # Create model metadata
        model_info = {
            "id": model_id,
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "parameters": params,
            "description": description,
            "path": new_model_path
        }
        
        # Update registry
        metadata = self._load_metadata()
        metadata["models"].append(model_info)
        self._save_metadata(metadata)
        
        logger.info(f"Saved model checkpoint: {model_id}")
        return model_id
    
    def get_best_model(self, 
                      metric: str = "validation_loss", 
                      higher_is_better: bool = False) -> Optional[Dict]:
        """
        Get the best model based on a specific metric.
        """
        metadata = self._load_metadata()
        if not metadata["models"]:
            return None
            
        models = metadata["models"]
        best_model = max(models, 
                        key=lambda x: x["metrics"].get(metric, float('-inf')) if higher_is_better 
                        else -x["metrics"].get(metric, float('inf')))
        
        return best_model
    
    def set_current_model(self, model_id: str) -> None:
        """Set the current production model."""
        metadata = self._load_metadata()
        if not any(m["id"] == model_id for m in metadata["models"]):
            raise ValueError(f"Model {model_id} not found")
            
        metadata["current_model"] = model_id
        self._save_metadata(metadata)
        logger.info(f"Set current model to: {model_id}")
    
    def get_current_model(self) -> Optional[Dict]:
        """Get the current production model info."""
        metadata = self._load_metadata()
        current_id = metadata["current_model"]
        
        if not current_id:
            return None
            
        for model in metadata["models"]:
            if model["id"] == current_id:
                return model
                
        return None
    
    def list_models(self, 
                   limit: int = None, 
                   metric_threshold: Optional[Dict[str, float]] = None) -> list:
        """
        List all models, optionally filtered by metrics.
        
        Args:
            limit: Maximum number of models to return (newest first)
            metric_threshold: Dict of metric names and their minimum/maximum values
                e.g. {"validation_loss": {"max": 0.1}, "accuracy": {"min": 0.95}}
        """
        metadata = self._load_metadata()
        models = metadata["models"]
        
        # Filter by metrics if specified
        if metric_threshold:
            filtered_models = []
            for model in models:
                meets_criteria = True
                for metric, criteria in metric_threshold.items():
                    if "min" in criteria and model["metrics"].get(metric, float('-inf')) < criteria["min"]:
                        meets_criteria = False
                    if "max" in criteria and model["metrics"].get(metric, float('inf')) > criteria["max"]:
                        meets_criteria = False
                if meets_criteria:
                    filtered_models.append(model)
            models = filtered_models
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        if limit:
            models = models[:limit]
            
        return models

# Create singleton instance
model_manager = SimpleModelManager()