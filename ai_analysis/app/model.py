import tensorflow as tf
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibrationAutoencoder(tf.keras.Model):
    """
    Autoencoder model for vibration analysis that can be reconstructed from tensor details.
    Architecture is automatically determined from the tensor details of a trained model.
    """
    
    def __init__(self, tensor_details: List[Dict[str, Any]]):
        """
        Initialize the autoencoder model based on tensor details.
        
        Args:
            tensor_details: List of dictionaries containing tensor information from the TFLite model
        """
        super(VibrationAutoencoder, self).__init__()
        
        # Extract layer dimensions from tensor details
        self.layer_dims = self._extract_layer_dimensions(tensor_details)
        logger.info(f"Reconstructed layer dimensions: {self.layer_dims}")
        
        # Build encoder layers
        self.encoder_layers = []
        for i in range(len(self.layer_dims) - 1):
            if i < len(self.layer_dims) // 2:  # Only build encoder part
                self.encoder_layers.append(tf.keras.layers.Dense(
                    units=self.layer_dims[i + 1],
                    activation='relu' if i < len(self.layer_dims) // 2 - 1 else None,
                    name=f'encoder_layer_{i}'
                ))
        
        # Build decoder layers
        self.decoder_layers = []
        for i in range(len(self.layer_dims) // 2, len(self.layer_dims) - 1):
            self.decoder_layers.append(tf.keras.layers.Dense(
                units=self.layer_dims[i + 1],
                activation='relu' if i < len(self.layer_dims) - 2 else None,
                name=f'decoder_layer_{i}'
            ))
        
        # Input reshaping layers
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape((self.layer_dims[0], 1))

    def _extract_layer_dimensions(self, tensor_details: List[Dict[str, Any]]) -> List[int]:
        """
        Extract the layer dimensions from tensor details.
        
        Args:
            tensor_details: List of dictionaries containing tensor information
            
        Returns:
            List of layer dimensions in order
        """
        # First layer dimension (input)
        logger.info(f"Extracting dimensions from tensor details: {tensor_details}")
        
        input_shape = next(
            detail['shape'] for detail in tensor_details 
            if detail['name'] == 'input'
        )
        logger.info(f"Found input shape: {input_shape}")
        
        dimensions = [input_shape[1]]  # Get the sequence length (128)
        logger.info(f"Initial dimensions list: {dimensions}")
        
        # Extract intermediate layer dimensions from matmul operations
        matmul_layers = sorted(
            [detail for detail in tensor_details if 'MatMul' in detail['name']],
            key=lambda x: x['name']
        )
        logger.info(f"Found MatMul layers: {matmul_layers}")
        
        for layer in matmul_layers:
            output_dim = layer['shape'][1]
            logger.info(f"Processing layer {layer['name']} with output dim {output_dim}")
            if output_dim not in dimensions:
                dimensions.append(output_dim)
                logger.info(f"Added dimension {output_dim}, current dimensions: {dimensions}")
        
        logger.info(f"Final layer dimensions: {dimensions}")
        return dimensions

    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, 1)
            training: Boolean indicating if in training mode
            
        Returns:
            Reconstructed output tensor of shape (batch_size, sequence_length, 1)
        """
        # Flatten input
        x = self.flatten(inputs)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Reshape output back to original dimensions
        x = self.reshape(x)
        
        return x

    def get_config(self):
        """Return model configuration"""
        config = super(VibrationAutoencoder, self).get_config()
        config.update({
            'layer_dims': self.layer_dims
        })
        return config

def create_model_from_tensor_details(tensor_details: List[Dict[str, Any]]) -> VibrationAutoencoder:
    """
    Factory function to create a new model instance from tensor details.
    
    Args:
        tensor_details: List of dictionaries containing tensor information
        
    Returns:
        Initialized VibrationAutoencoder model
    """
    model = VibrationAutoencoder(tensor_details)
    
    # Build model with sample input
    sample_input = tf.keras.Input(shape=(128, 1))
    model(sample_input)
    
    logger.info("Created model with architecture:")
    model.summary(print_fn=logger.info)
    
    return model
