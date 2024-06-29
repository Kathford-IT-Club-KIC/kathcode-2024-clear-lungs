from gradcam_VGG19 import * # For pth extension
from gradcam_VGG19_tf import * # For h5 extension
 
# For VGG19 
model_path = r'./super.pth'
image_path = r"./normal_two.jpg"
main_vgg19(model_path, image_path)

# For VGG19 TF
# model_path = r'./super.h5'
# image_path = r"./pneumonia.jpeg"
# main_vgg19_tf(model_path, image_path)

