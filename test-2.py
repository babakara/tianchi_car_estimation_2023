from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import models, transforms

#!pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

import os
from PIL import Image

import cv2

import json
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

cls_one = ['clear_weather', 'cloudy', 'foggy', 'rainning', 'snowy']
cls_two = ['city_road', 'expressway', 'gas_station', 'parking', 'suburbs', 'tunnel']

# 使用预训练的 ResNet-50 模型
model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(cls_one))
model_first = model_ft
model_first = model_first.to(device)

folder_first = './classification_models/weather'
model_filename = 'resnet_50_in418_out5_ft.pth'

model_path = os.path.join(folder_first, model_filename)
print(model_path)
model_first.load_state_dict(torch.load(model_path))
model_first.eval()


def classification(image, model, class_names, device):
    # Make a prediction

    if image is not None:
        # Make predictions
        transform = transforms.Compose([
        transforms.Resize((418, 418)),  # 根据模型的输入大小调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # 添加一个批次维度
        input_batch = input_batch.to(device)
        
        # 使用模型进行推理
        with torch.no_grad():
            
            output = model(input_batch)

            # Convert the output to probabilities using softmax
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Get the predicted class index
            predicted_class_index = torch.argmax(probabilities).item()

            # Print the predicted class and probability
            predicted_class_name = class_names[predicted_class_index]
            predicted_probability = probabilities[predicted_class_index].item()

            print(f"Predicted class: {predicted_class_name}")
            print(f"Probability: {predicted_probability:.2%}")
            return predicted_class_index, predicted_class_name, predicted_probability
    else:
        print("Image is None. Check if the image was successfully opened.")
        return False, False, 0
    
    
image = Image.open('./test_images/1.png')

predicted_class_index, predicted_class_name, predicted_probability = classification(image, model_first, cls_one, device)