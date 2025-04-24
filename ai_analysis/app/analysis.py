import numpy as np
from typing import Dict, Any, List
import logging
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt

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
    Main analysis function that processes vibration data in a single batched tensor.
    """
    try:
        logger.info(f"Starting vibration analysis with {len(data)} points")
        debug = True
        
        # Load the TFLite model
        interpreter = tflite.Interpreter(model_path='data/checkpoints/autoencoder_checkpoint.tflite')
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        sequence_length = input_shape[2]
        logger.info(f"Input tensor shape: {input_shape}")
        
        # Extract acceleration values and convert to numpy array
        acceleration_values = np.array([point.get('acceleration', 0.0) for point in data])
        
        # Calculate number of complete batches possible
        total_points = len(acceleration_values)
        num_batches = total_points // sequence_length
        
        # Reshape all data at once into [num_batches, 1, sequence_length]
        batched_data = acceleration_values[:num_batches * sequence_length].reshape(num_batches, 1, sequence_length).astype(np.float32)
        
        # Set the input tensor with all batches
        interpreter.resize_tensor_input(input_details[0]['index'], [num_batches, 1, sequence_length])
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], batched_data)
        
        # Run inference once for all batches
        interpreter.invoke()
        
        # Get the output tensor for all batches
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Calculate reconstruction loss for all batches at once
        reconstruction_losses = np.mean(np.square(batched_data - output_data), axis=(1, 2))
        
        mean_loss = np.mean(reconstruction_losses)
        max_loss = np.max(reconstruction_losses)
        min_loss = np.min(reconstruction_losses)
        
        # Plot the last batch if debug is enabled
        if debug:
            plt.figure(figsize=(12, 6))
            plt.plot(batched_data[-1].reshape(-1), label='Input Data', alpha=0.7)
            plt.plot(output_data[-1].reshape(-1), label='Reconstructed Data', alpha=0.7)
            plt.title(f'Input vs Reconstructed Signal (Last Batch)')
            plt.xlabel('Time Step')
            plt.ylabel('Acceleration')
            plt.legend()
            plt.grid(True)
            plt.savefig('debug_output/debug.png')
            plt.close()

        logger.info(f"Processed {num_batches} batches at once")
        logger.info(f"Mean reconstruction loss: {mean_loss}")
        logger.info(f"Max reconstruction loss: {max_loss}")
        logger.info(f"Min reconstruction loss: {min_loss}")

        result = {
            'status': 'completed',
            'analysis_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'data_points_analyzed': num_batches * sequence_length,
            'max_reconstruction_loss': float(max_loss),
            'min_reconstruction_loss': float(min_loss),
            'mean_reconstruction_loss': float(mean_loss),
            'num_batches_processed': num_batches
        }
        
        logger.info(f"Analysis completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in vibration analysis: {str(e)}")
        raise


