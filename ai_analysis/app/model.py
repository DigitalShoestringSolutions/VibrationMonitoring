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
        # Use Reshape layer with proper target shape
        self.reshape = tf.keras.layers.Reshape((self.layer_dims[-1], 1))
        
        # Create projection layer for shape adjustment
        self.projection = tf.keras.layers.Dense(self.layer_dims[-1], name='projection_layer')

    def _extract_layer_dimensions(self, tensor_details: List[Dict[str, Any]]) -> List[int]:
        """
        Extract the layer dimensions from tensor details.
        
        Args:
            tensor_details: List of dictionaries containing tensor information
            
        Returns:
            List of layer dimensions in order
        """
        # First layer dimension (input)
        input_shape = next(
            detail['shape'] for detail in tensor_details 
            if detail['name'] == 'input'
        )
        dimensions = [input_shape[2]]  # Get the sequence length (128)
        
        # Find all unique dimensions from MatMul operations
        matmul_layers = []
        for detail in tensor_details:
            if 'MatMul' in detail['name']:
                # Extract the layer number from the name
                name_parts = detail['name'].split('/')
                layer_num = next((part for part in name_parts if 'matmul' in part.lower()), '')
                if layer_num:
                    # Get the number from matmul_1, matmul_2 etc.
                    num = int(layer_num.split('_')[-1]) if '_' in layer_num else 0
                    matmul_layers.append((num, detail))
        
        # Sort by layer number to maintain order
        matmul_layers.sort(key=lambda x: x[0])
        
        # Build dimensions list
        seen_dims = set([dimensions[0]])
        for _, layer in matmul_layers:
            output_dim = layer['shape'][1]
            if output_dim not in seen_dims:
                dimensions.append(output_dim)
                seen_dims.add(output_dim)
        
        # Add final output dimension if needed
        if dimensions[0] not in seen_dims:
            dimensions.append(dimensions[0])
        
        logger.info(f"Extracted dimensions: {dimensions}")
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
        # Store original shape
        batch_size = tf.shape(inputs)[0]
        sequence_length = inputs.shape[1]
        
        # Flatten input
        x = self.flatten(inputs)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Ensure the output has the correct number of elements using TensorFlow operations
        output_size = tf.reduce_prod(tf.shape(x))
        target_size = batch_size * sequence_length
        x = tf.cond(
            tf.not_equal(output_size, target_size),
            lambda: self.projection(x),
            lambda: x
        )
        
        # Reshape output back to original dimensions
        x = tf.reshape(x, (batch_size, sequence_length, 1))
        
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
