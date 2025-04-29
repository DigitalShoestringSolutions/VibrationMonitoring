import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import logging
from .influx import influx_service
from .model_management import model_manager
from .model import VibrationAutoencoder, create_model_from_tensor_details
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
        
    # Calculate number of complete sequences possible
    total_points = len(acceleration_values)
    num_sequences = total_points // sequence_length
    
    # Reshape into sequences
    sequences = acceleration_values[:num_sequences * sequence_length].reshape(-1, sequence_length, 1)
    
    # Create dataset from pre-shaped sequences
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    logger.info(f"Dataset element spec: {dataset.element_spec}")
    logger.info(f"Created dataset with {num_sequences} sequences of shape {sequences.shape}")
    return dataset


def finetune_model(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fine-tune the autoencoder model using new data.
    
    Args:
        data: Dictionary containing:
            - start_time: ISO format start timestamp
            - end_time: ISO format end timestamp
            
    Returns:
        Dictionary containing training results
    """

    debug = False
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
        logger.info(f"Tensor details: {tensor_details}")

        # get sequence length from tensor details
        sequence_length = input_details[0]['shape'][2]
        logger.info(f"Sequence length: {sequence_length}")

        # Create model from tensor details
        model = create_model_from_tensor_details(tensor_details)

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

        logger.info("Successfully loaded weights from current model")

        # Prepare Data for Training
        dataset = prepare_training_data(data, sequence_length)


        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(
            dataset,
            epochs=10
        )

        # Log training metrics
        logger.info("Training metrics:")
        for epoch, loss in enumerate(history.history['loss']):
            logger.info(f"Epoch {epoch + 1}: loss = {loss:.4f}")
        
        final_loss = history.history['loss'][-1]
        logger.info(f"Final training loss: {final_loss:.4f}")





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
                'training_loss': final_loss,
            },
            params={
                'sequence_length': sequence_length,
                'epochs': 10
            },
            description="First attempt at fine-tuning model"
        )
        # Clean up temporary file
        os.remove(temp_model_path)
        logger.info(f"Saved model with ID: {model_id}")

        




        
        # # Query training data
        # training_data = influx_service.query_vibration_data(
        #     start_time=data['start_time'],
        #     end_time=data['end_time']
        # )
        
        # if not training_data:
        #     raise ValueError("No training data available for the specified time range")
            
        # # Prepare data
        # X_train = prepare_training_data(training_data)
        
        # Fine-tune model
        # history = model.fit(
        #     X_train,
        #     X_train,
        #     epochs=10,
        #     batch_size=32,
        #     validation_split=0.2,
        #     verbose=1
        # )
        
        # # Save fine-tuned model
        # metrics = {
        #     'final_loss': history.history['loss'][-1],
        #     'final_val_loss': history.history['val_loss'][-1]
        # }
        
        # model_id = model_manager.save_model(
        #     model_path=model_path,
        #     metrics=metrics,
        #     params={
        #         'epochs': 10,
        #         'batch_size': 32,
        #         'sequence_length': 128
        #     },
        #     description="Fine-tuned model"
        # )
        
        # result = {
        #     'status': 'completed',
        #     'model_id': model_id,
        #     'metrics': metrics,
        #     'training_samples': len(X_train)
        # }
        
        # logger.info(f"Model fine-tuning completed: {result}")
        # return result
        
    except Exception as e:
        logger.error(f"Error in model fine-tuning: {str(e)}")
        raise 