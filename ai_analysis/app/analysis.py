import numpy as np
from typing import Dict, Any, List
import logging
import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def calculate_rms(acceleration: List[float]) -> float:
    """
    Calculate Root Mean Square of acceleration data.
    
    Args:
        acceleration: List of acceleration values
        
    Returns:
        RMS value
    """
    # return np.sqrt(np.mean(np.square(acceleration)))
    return 1
def detect_anomaly(rms_acceleration: float, temperature: float) -> bool:
    """
    Detect if the current readings indicate an anomaly.
    
    Args:
        rms_acceleration: RMS acceleration value
        temperature: Temperature reading
        
    Returns:
        True if anomaly detected, False otherwise
    """
    # Example thresholds - adjust based on your requirements
    ACCELERATION_THRESHOLD = 5.0
    TEMPERATURE_THRESHOLD = 80.0
    
    # return (rms_acceleration > ACCELERATION_THRESHOLD or 
    #         temperature > TEMPERATURE_THRESHOLD)
    return False
def analyze_vibration_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main analysis function that processes vibration data.
     - takes the influx data across the time period
     - load the data with a dataloader

     - load the model checkpoint
     - run inference on the data in batches
     - calculate the reconsstruction loss across the batch
     - return max reconstruction loss, min and mean reconstruction loss
    
    Args:
        Data from influxdb, what is the formt though?
    Returns:
        Dictionary containing analysis results
    """
    try:
        logger.info(f"Starting vibration analysis with {len(data)} points")
        
        # Extract acceleration values and convert to numpy array
        acceleration_values = np.array([point.get('acceleration', 0.0) for point in data])
        
        # Ensure we have the right number of data points (pad or truncate to 128)
        if len(acceleration_values) > 128:
            acceleration_values = acceleration_values[:128]
        elif len(acceleration_values) < 128:
            # Pad with zeros if we have less than 128 points
            padding = np.zeros(128 - len(acceleration_values))
            acceleration_values = np.concatenate([acceleration_values, padding])
        
        # Reshape to match the model's expected input shape [1, 1, 128]
        model_input = acceleration_values.reshape(1, 1, 128).astype(np.float32)
        
        # Load and run the TFLite model
        interpreter = tflite.Interpreter(model_path='data/checkpoints/autoencoder_checkpoint.tflite')
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], model_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        reconstruction_loss = np.mean(np.square(model_input - output_data))
        logger.info(f"Raw reconstruction loss: {reconstruction_loss}")

        result = {
            'status': 'completed',
            'analysis_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'data_points_analyzed': len(data),
            'max_reconstruction_loss': 0.0,
            'min_reconstruction_loss': 0.0,
            'mean_reconstruction_loss': float(reconstruction_loss)
        }
        
        logger.info(f"Analysis completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in vibration analysis: {str(e)}")
        raise


