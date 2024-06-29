import torch
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

random.seed(a=None, version=2)

def preprocess_image(img_path, mean, std, H=320, W=320):
    preprocess = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image = Image.open(img_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    return image

def get_mean_std(image_path):
    sample_image = Image.open(image_path).convert('RGB')
    sample_image = np.array(sample_image.resize((320, 320)))
    mean = np.mean(sample_image, axis=(0, 1)) / 255.0
    std = np.std(sample_image, axis=(0, 1)) / 255.0
    return mean, std

def grad_cam(model, img_tensor, target_class, target_layer):
    model.eval()
    
    # Register hooks
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_module = dict(model.named_modules()).get(target_layer)
    if target_module is None:
        raise ValueError(f"Layer {target_layer} not found in the model")
    
    handle_forward = target_module.register_forward_hook(forward_hook)
    handle_backward = target_module.register_backward_hook(backward_hook)
    
    # Forward pass
    output = model(img_tensor)
    target = output[0, target_class]
    
    # Backward pass
    model.zero_grad()
    target.backward()

    # Get the gradients and activations
    gradients = gradients[0].cpu().data.numpy()
    activations = activations[0].cpu().data.numpy()

    # Calculate the weights
    weights = np.mean(gradients, axis=(2, 3))
    cam = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)[0]

    # Process CAM
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (320, 320))
    return cam

def plot_gradcam(original_img, gradcam, label, prediction, alpha=0.5):
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(original_img)

    plt.subplot(1, 2, 2)
    plt.title(f"{label}: p={prediction:.3f}")
    plt.axis('off')
    plt.imshow(original_img)
    plt.imshow(gradcam, cmap='jet', alpha=alpha)
    plt.show()

def main_vgg19(model_path, image_path):
    # Load the model
    model = models.vgg19(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)  # Assuming binary classification
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Print model layers to identify the target layer
    print(dict(model.named_modules()).keys())
    
    # Image path
    image_path = image_path 

    # Get mean and std
    mean, std = get_mean_std(image_path)

    # Preprocess the image
    img_tensor = preprocess_image(image_path, mean, std)

    # Forward pass and get the predicted class index
    output = model(img_tensor)
    _, class_idx = torch.max(output, 1)
    class_idx = class_idx.item()
    prediction = F.softmax(output, dim=1)[0][class_idx].item()

    # Generate GradCAM
    gradcam = grad_cam(model, img_tensor, class_idx, target_layer='features.34')  # Ensure this is the correct layer

    # Load the original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (320, 320))

    # Plot GradCAM
    label = 'Pneumonia' if class_idx == 1 else 'Normal'
    plot_gradcam(original_img, gradcam, label, prediction)


