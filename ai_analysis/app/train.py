import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import logging
from .influx import influx_service
from .model_management import model_manager
from .model import DynamicAutoencoder, StaticAutoencoder, create_model_from_tensor_details
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data(data: Dict[str, Any], sequence_length: int) -> np.ndarray:
    """
    Prepare vibration data for training.
    
    Args:
        data: Dictionary influx data 
        sequence_length: Length of each sequence
        
    Returns:
        Numpy array of prepared training data in batch
    """
    acceleration_values = np.array([point.get('acceleration', 0.0) for point in data])

    acceleration_values = acceleration_values - np.mean(acceleration_values) # Center the data around 0
    # Calculate number of complete sequences possible
    total_points = len(acceleration_values)
    num_sequences = total_points // sequence_length
    
    # Reshape into sequences
    sequences = acceleration_values[:num_sequences * sequence_length].reshape(-1, sequence_length, 1)
    
    # Create dataset from pre-shaped sequences
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.map(lambda x: (x, x))  # Return same data for both input and target
    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"Dataset element spec: {dataset.element_spec}")
    logger.info(f"Created dataset with {num_sequences} sequences of shape {sequences.shape}")
    return dataset


def finetune_model(data: Dict[str, Any], start_time: str, end_time: str) -> Dict[str, Any]:
    """
    Fine-tune the autoencoder model using new data.
    
    Args:
        data: Dictionary containing:
            - start_time: ISO format start timestamp
            - end_time: ISO format end timestamp
            
    Returns:
        Dictionary containing training results
    """

    debug = True
    try:
        logger.info("Starting model fine-tuning")
        
        # Get current model
        current_model = model_manager.get_current_model()
        if not current_model:
            raise ValueError("No current model available for fine-tuning")
            
        # Load the current model
        model_path = current_model['path']
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()

        logger.info(f"Input details: {input_details}")
        logger.info(f"Output details: {output_details}")
        # logger.info(f"Tensor details: {tensor_details}")

        # get sequence length from tensor details
        shape = input_details[0]['shape']
        sequence_length = shape[3] if len(shape) == 4 else shape[2]
        logger.info(f"Sequence length: {sequence_length}")

        # Create model from tensor details
        # model = create_model_from_tensor_details(tensor_details) # for dynamic model
        model = StaticAutoencoder()
        
        # Load weights from the current model
        # First, create a dummy input to build the model
        sample_input = tf.keras.Input(shape=(1,1,sequence_length))
        model(sample_input)
        
        # Extract weights from TFLite model
        for detail in tensor_details:
            if 'kernel' in detail['name'].lower() or 'bias' in detail['name'].lower():
                tensor = interpreter.get_tensor(detail['index'])
                # Find corresponding layer in our model
                layer_name = detail['name'].split('/')[0]  # Get base layer name
                for layer in model.layers:
                    if layer_name in layer.name:
                        if 'kernel' in detail['name'].lower():
                            layer.kernel.assign(tensor)
                        elif 'bias' in detail['name'].lower():
                            layer.bias.assign(tensor)

        logger.info(f"Successfully loaded weights from current model: {current_model['id']}")
    
        # Prepare Data for Training
        dataset = prepare_training_data(data, sequence_length)
        epochs = 30

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(
            dataset,
            epochs=epochs
        )

        # Log training metrics
        logger.info("Training metrics:")
        for epoch, loss in enumerate(history.history['loss']):
            logger.info(f"Epoch {epoch + 1}: loss = {loss:.4f}")
        
        final_loss = history.history['loss'][-1]
        logger.info(f"Final training loss: {final_loss:.4f}")

        # Debug Plots
        if debug:
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot 1: Training Loss
            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            # Plot 2: Sample Input/Output
            # Get a sample batch from dataset
            acceleration_values = np.array([point.get('acceleration', 0.0) for point in data])
            x = acceleration_values[:128]
            x = tf.reshape(x, (1, 128, 1))  # Reshape to match model input shape (batch_size, sequence_length, features)
            # Get model prediction
            logger.info(f"x shape: {x.shape}")
            y_pred = model(x)
            
            # Plot sample input and output
            sample_idx = 0  # Take first sample from batch
            ax2.plot(x[sample_idx].numpy().flatten(), label='Input', alpha=0.7)
            ax2.plot(y_pred[sample_idx].numpy().flatten(), label='Model Output', alpha=0.7)
            ax2.set_title('Input vs Model Reconstruction')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Amplitude')
            ax2.legend()

            plt.tight_layout()
            
            # Save debug plot to file
            debug_output_dir = 'debug_output'
            os.makedirs(debug_output_dir, exist_ok=True)
            plot_path = os.path.join(debug_output_dir, 'model_analysis.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved debug plots to {plot_path}")



        # Save the model
        temp_model_path = 'data/checkpoints/temp_model.tflite'
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(temp_model_path, 'wb') as f:
            f.write(tflite_model)
        # Save model with metadata using model manager
        model_id = model_manager.save_model(
            model_path=temp_model_path,
            metrics={
                'training_loss': str(final_loss),
            },
            params={
                'sequence_length': str(sequence_length),
                'epochs': str(epochs),
                'start_time': start_time,
                'end_time': end_time,
                'previous_model_id': current_model['id']
            },
            description="First attempt at fine-tuning model"
        )
        # Clean up temporary file
        os.remove(temp_model_path)
        logger.info(f"Saved model with ID: {model_id}")

        # Update the current model in the model manager
        model_manager.set_current_model(model_id)

        return {
            'status': 'completed',
            'message': 'Model fine-tuned successfully',
            'model_id': model_id
        }

    except Exception as e:
        logger.error(f"Error in model fine-tuning: {str(e)}")
        raise 