import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Load the Keras model
keras_model = tf.keras.models.load_model('./super.h5')

# Extract the weights from the Keras model
keras_weights = keras_model.get_weights()

# Define the equivalent PyTorch model
pytorch_model = models.vgg19(pretrained=False)
pytorch_model.classifier[6] = nn.Linear(in_features=4096, out_features=2)  # Modify based on your Keras model's output layer

def transfer_weights(keras_weights, pytorch_model):
    keras_index = 0
    for pytorch_layer in pytorch_model.children():
        if isinstance(pytorch_layer, nn.Conv2d):
            pytorch_layer.weight.data = torch.tensor(keras_weights[keras_index].transpose([3, 2, 0, 1]))
            keras_index += 1
            pytorch_layer.bias.data = torch.tensor(keras_weights[keras_index])
            keras_index += 1
        elif isinstance(pytorch_layer, nn.Linear):
            pytorch_layer.weight.data = torch.tensor(keras_weights[keras_index].T)
            keras_index += 1
            pytorch_layer.bias.data = torch.tensor(keras_weights[keras_index])
            keras_index += 1

# Transfer the weights
transfer_weights(keras_weights, pytorch_model)

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), './super.pth')
