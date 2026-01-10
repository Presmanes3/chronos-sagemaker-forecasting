"""
ChronosPredictorService - Encapsulates AutoGluon TimeSeriesPredictor.
"""
import os
import shutil
import tempfile
from typing import Dict, Any, Optional

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from logger_config import setup_logger


logger = setup_logger()


class ChronosPredictorService:
    """Encapsulates AutoGluon TimeSeriesPredictor with validation and metadata access."""
    
    def __init__(self):
        self.predictor: Optional[TimeSeriesPredictor] = None
        self.model_path: Optional[str] = None
        self._tmp_model_dir: Optional[str] = None
    
    def load_model(self, model_dir: str) -> None:
        """Load AutoGluon model from read-only directory to writable /tmp."""
        try:
            logger.info(
                "Loading model from read-only directory",
                extra={"request_id": "startup", "model_path": model_dir}
            )
            
            # Verify the directory exists
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
            
            # Verify it contains required AutoGluon files
            required_files = ["predictor.pkl"]
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
            if missing_files:
                logger.error(
                    f"Missing required model files: {missing_files}",
                    extra={"request_id": "startup", "model_path": model_dir}
                )
                logger.error(
                    f"Contents of {model_dir}:",
                    extra={"request_id": "startup"}
                )
                for item in os.listdir(model_dir):
                    logger.error(
                        f"  - {item}",
                        extra={"request_id": "startup"}
                    )
                raise FileNotFoundError(f"Model directory missing required files: {missing_files}")
            
            # Copy model to writable /tmp directory (required by AutoGluon)
            self._tmp_model_dir = os.path.join(tempfile.gettempdir(), "chronos_model_copy")
            if os.path.exists(self._tmp_model_dir):
                shutil.rmtree(self._tmp_model_dir)
            shutil.copytree(model_dir, self._tmp_model_dir)
            
            logger.info(
                "Model copied to writable directory",
                extra={"request_id": "startup", "model_path": self._tmp_model_dir}
            )
            
            # Load predictor from /tmp
            self.predictor = TimeSeriesPredictor.load(self._tmp_model_dir)
            
            # Redirect internal paths to /tmp (critical for AutoGluon)
            self.predictor.path = self._tmp_model_dir
            self.predictor._learner.path = self._tmp_model_dir
            self.predictor._trainer.path = os.path.join(self._tmp_model_dir, "models")
            
            self.model_path = self._tmp_model_dir
            
            logger.info(
                "Model loaded successfully",
                extra={
                    "request_id": "startup",
                    "model_path": self.model_path,
                    "prediction_length": self.prediction_length,
                    "target": self.target
                }
            )
            
        except Exception as e:
            logger.error(
                "Model loading failed",
                extra={"request_id": "startup", "model_path": model_dir, "error": str(e)},
                exc_info=True
            )
            raise RuntimeError(f"Failed to load model: {e}")
    
    @property
    def prediction_length(self) -> int:
        """Get model's forecast horizon."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        return self.predictor.prediction_length
    
    @property
    def target(self) -> str:
        """Get model's target variable name."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        return self.predictor.target
    
    @property
    def eval_metric(self) -> Optional[str]:
        """Get model's evaluation metric."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        return getattr(self.predictor, "eval_metric", None)
    
    def validate_input(self, ts_df: TimeSeriesDataFrame, request_id: str) -> None:
        """Validate input data against model requirements."""
        # Check required columns
        if self.target not in ts_df.columns:
            raise ValueError(f"Input must contain target column: '{self.target}'")
        
        # Check minimum data points (at least 1 observation per series)
        min_length = ts_df.groupby(level=0).size().min()
        if min_length < 1:
            raise ValueError("Each series must contain at least 1 observation")
        
        logger.info(
            "Input validation passed",
            extra={
                "request_id": request_id,
                "model_path": self.model_path,
                "num_series": ts_df.num_items,
                "min_length": int(min_length)
            }
        )
    
    def predict(self, ts_df: TimeSeriesDataFrame, request_id: str) -> TimeSeriesDataFrame:
        """Execute prediction with proper working directory management."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Validate input
            self.validate_input(ts_df, request_id)
            
            # Create temporary working directory (required by AutoGluon)
            tmp_dir = tempfile.mkdtemp(dir="/tmp")
            original_cwd = os.getcwd()
            
            try:
                os.chdir(tmp_dir)
                
                logger.info(
                    "Starting prediction",
                    extra={
                        "request_id": request_id,
                        "model_path": self.model_path,
                        "num_series": ts_df.num_items,
                        "prediction_length": self.prediction_length
                    }
                )
                
                predictions = self.predictor.predict(ts_df)
                
                logger.info(
                    "Prediction completed successfully",
                    extra={
                        "request_id": request_id,
                        "model_path": self.model_path,
                        "num_predictions": len(predictions)
                    }
                )
                
                return predictions
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(
                "Prediction failed",
                extra={"request_id": request_id, "model_path": self.model_path, "error": str(e)},
                exc_info=True
            )
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        if self.predictor is None:
            raise RuntimeError("Model not loaded")
        
        return {
            "prediction_length": self.prediction_length,
            "target": self.target,
            "eval_metric": self.eval_metric,
            "model_path": self.model_path
        }
