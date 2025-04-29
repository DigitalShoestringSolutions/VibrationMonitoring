import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import logging
from .influx import influx_service
from .model_management import model_manager
from .model import VibrationAutoencoder, create_model_from_tensor_details


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_training_data(data: Dict[str, Any]) -> np.ndarray:
    """
    Prepare vibration data for training.
    
    Args:
        data: Dictionary containing vibration data points
        
    Returns:
        Numpy array of prepared training data
    """
    return

def finetune_model(model, train_data, val_data, epochs=10, batch_size=32):
    """
    Fine-tune the model on the training data.
    
    Args:
        model: The pre-trained model
        train_data: Training dataset
        val_data: Validation dataset
        epochs: Number of epochs to train
        batch_size: Batch size for training
        
    Returns:
        Fine-tuned model
    """
    # Create dummy input for model compilation
    sample_input = tf.random.normal((batch_size, 128, 1))
    
    # Verify model output shape
    sample_output = model(sample_input)
    logger.info(f"Model input shape: {sample_input.shape}")
    logger.info(f"Model output shape: {sample_output.shape}")
    
    if sample_input.shape != sample_output.shape:
        raise ValueError(f"Model input shape {sample_input.shape} does not match output shape {sample_output.shape}")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    return model, history

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
        logger.info(f"Tensor details: {tensor_details}")



        model = create_model_from_tensor_details(tensor_details)

        # Load weights from the current model
        # First, create a dummy input to build the model
        sample_input = tf.keras.Input(shape=(1,1,128))
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


        if debug:
            # Create sample sine wave input
            t = np.linspace(0, 2*np.pi, 128)
            sample_data = np.sin(t).reshape(-1, 128, 1)
            logger.info(f"Sample data shape: {sample_data.shape}")
            output_before = model(sample_data)
            
            # Create figure with two subplots
            plt.figure(figsize=(12, 6))
            
            # Plot input vs output before weight loading
            plt.subplot(2, 1, 1)
            plt.plot(t, sample_data.squeeze(), label='Input')
            plt.plot(t, output_before.numpy().squeeze(), label='Output')
            plt.title('Model Input vs Output Before Loading Weights')
            plt.legend()
            plt.grid(True)
            
            # Plot input vs output after weight loading
            plt.subplot(2, 1, 2)
            output_after = model(sample_data)
            plt.plot(t, sample_data.squeeze(), label='Input')
            plt.plot(t, output_after.numpy().squeeze(), label='Output')
            plt.title('Model Input vs Output After Loading Weights')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('debug_output/model_io_comparison.png')
            plt.close()
            
            logger.info("Saved input/output comparison plot to model_io_comparison.png")


        




        
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