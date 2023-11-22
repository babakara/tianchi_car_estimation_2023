from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

import cv2

import json
from datetime import datetime

# 扫描文件夹, 视频文件名称存入list
# video转frame
# frame分类、目标检测
# 距离估计
# 行为估计
# 合并结果

# 写入json文件



def read_model():
    # 读取不同的分类模型
    dict_class_names = dict()
    dict_class_names['weather'] = ['clear_weather', 'cloudy', 'foggy', 'rainning', 'snowy']
    dict_class_names['abnormal_condition'] = ['cracked', 'normal', 'standing_water', 'uneven', 'water']
    dict_class_names['scerario'] = ['city_road', 'expressway', 'gas_station', 'parking', 'suburbs', 'tunnel']
    dict_class_names['road_structure'] = ['T-junction', 'crossroad', 'lane_merging', 'normal_road', 'parking-lot', 'ramp', 'round-about']
    dict_class_names['period'] = ['dawn-or-dusk', 'daytime', 'night', 'unknown']
    
    dict_class_names['general_obstacle'] = ['nothing', 'speed bumper', 'traffic cone', 'water horse', 'stone', 'manhole cover', 'unknown']
    dict_class_names['closest_participants_type'] = ['passenger car', 'bus', 'truck', 'pedestrain', 'policeman', 'nothing']
    dict_class_names['ego_car_behavior'] = ['slow down', 'go straight', 'turn right', 'turn left', 'stop', 'U-turn', 'speed up', 'lane change', 'others']
    dict_class_names['closest_participants_behavior'] = ['slow down', 'go straight', 'turn right', 'turn left', 'stop', 'U-turn', 'speed up', 'lane change' ,'others']
    # 在补充完整后面三个类之前不要运行脚本。
    
    dict_model = dict()
    
    dict_dir_path = dict()
    dict_dir_path['period'] = './classification_models/daytime'
    dict_dir_path['weather'] = './classification_models/weather'
    dict_dir_path['scerario'] = './classification_models/scerario'
    dict_dir_path['road_structure'] = './classification_models/roadStructure'
    dict_dir_path['abnormal_condition'] = './classification_models/abnormal_road'
    
    dict_dir_path['general_obstacle'] = './detection_models/detection_obstacle'
    dict_dir_path['closest_participants_type'] = './detection_models/closest_participants_type'
    dict_dir_path['ego_car_behavior'] = './detection_models/ego_car_behavior'
    dict_dir_path['closest_participants_behavior'] = './detection_models/closest_participants_behavior'
    
    model_filename = 'resnet_50_in418_out5_ft.pth'
    
    yolo_model_filename = 'best.pt'

    # from a github repo load pytorch base model
    repo = 'pytorch/vision'
    model_ft = torch.hub.load(repo, 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features

    # 不同的模型有不同的class_names
    for key, value in dict_class_names.items():
        
        if key in ['period', 'weather', 'scerario', 'road_structure', 'abnormal_condition']:
            # classification
            model_temp = model_ft
            model_temp.fc = nn.Linear(num_ftrs, len(value))
            dict_model[key] = model_temp
            dict_model[key] = dict_model[key].to(device)

            model_path = os.path.join(dict_dir_path[key], model_filename)
            dict_model[key].load_state_dict(torch.load(model_path))
            dict_model[key].eval()
            
        elif key in ['general_obstacle']:
            # object detection and distance estimation
            model_det = YOLO(os.path.join(dict_dir_path[key], yolo_model_filename))  # load a custom model
            dict_model[key] = model_det
            
    return dict_model, dict_class_names

# 读取video
def find_mp4_files(folder_path):
    mp4_files = []
    for root, dirs, files in os.walk(folder_path):
        mp4_files.extend([file for file in files if file.lower().endswith('.mp4')])
    return mp4_files


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

# 检测路上的目标
def object_detect(image, model):
    # Make predictions with the loaded model
    results = model(image)
    
    # 查找是否有目标，没有就返回nothing
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        probs = result.probs  # Probs object for classification outputs
        
    # 根据焦距和预先知识计算到相机的距离
    
    # 展示结果
    # for r in results:
    #     im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
    #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
    #     im.show()  # 显示图像 im.show()
    #     im.save('results.jpg')  # 保存图像
    
    return results


# 并从文件名中提取点之前的部分
def extract_prefix(filename):
    base_name = os.path.basename(filename)
    prefix, _ = os.path.splitext(base_name)
    return prefix

# 获取当前时间的YYMMDD
def get_yymmdd():
    # 获取当前日期和时间
    now = datetime.now()

    # 格式化日期为YYMMDD
    yymmdd = now.strftime("%y%m%d")

    return yymmdd

if __name__ == '__main__':
    # find device, cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 加载pytorch resnet50分类模型
    models, classes = read_model()
    # 加载待检测的MP4文件
    mp4_folder = 'F:\Video\Tianchi_combo/test_video_mp4'
    
    output_folder = 'F:\Video\Tianchi_combo/test_video_output'
    os.makedirs(output_folder, exist_ok=True)
    list_mp4 = find_mp4_files(mp4_folder)
    
    data = {'author':'abc', 'time': get_yymmdd(), 'model':'model_name', 'test_results':[]}

    for mp4 in list_mp4:
        index = 1
        # 视频名字提取
        video_name = extract_prefix(mp4)
        data_piece = {'clip_id': video_name,
            'scerario':'cityroad',
            'weather':'unknown',
            'period':'night',
            'road_structure':'ramp',
            'general_obstacle':'nothing',
            'abnormal_condition':'nothing',
            'ego_car_behavior':'turning right',
            'closest_participants_type':'passenger car',
            'closest_participants_behavior':'braking'
            }
        mp4_file_path = os.path.join(mp4_folder, mp4)
        print(mp4_file_path)
        try:
            extract_freq = 4*30 #间隔视频帧, 一个视频取一帧足够了

            cap = cv2.VideoCapture()
            if not cap.open(mp4_file_path):
                print('fail open mp4 file')
                exit(1)
            count = 1
            while True:
                _,frame = cap.read()
                if frame is None:
                    break
                if count % extract_freq == 0:
                    # 分类任务
                    # Convert the OpenCV BGR image to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert the NumPy array to a PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    for key, model in models.items():
                        # 预测类别，概率
                        predicted_class_index, predicted_class_name, predicted_probability = classification(pil_image, model, classes[key], device)
                        
                        # 名词修改
                        if key == 'weather':
                            if predicted_class_name == 'clear_weather':
                                predicted_class_name = 'clear'
                            
                        if key == 'abnormal_condition':
                            if predicted_class_name == 'normal':
                                predicted_class_name = 'nothing'
                            elif predicted_class_name == 'water':
                                predicted_class_name = 'oil or water stain'
                            elif predicted_class_name == 'standing_water':
                                predicted_class_name = 'standing water'
                            
                        if key == 'period':
                            if predicted_class_name == 'dawn-or-dusk':
                                predicted_class_name = 'dawn or dusk'
                                
                        if key == 'scerario':
                            if predicted_class_name == 'city_road':
                                predicted_class_name = 'city street'
                            elif predicted_class_name == 'parking':
                                predicted_class_name = 'parking-lot'
                            elif predicted_class_name == 'gas_station':
                                predicted_class_name = 'gas or charging stations'
                        
                        data_piece[key] = predicted_class_name
                        
                    index += 1
                count += 1

            cap.release()
            # 一个视频处理完成，合并data_piece到data中
            data['test_results'].append(data_piece)
            
        except Exception as e:
            print(f"提取帧失败：{mp4_file_path}\n错误信息：{e}")
    # 所有视频处理完成，保存json文件
    with open(os.path.join(output_folder, 'test_results.json'), 'w') as f:
        json.dump(data, f, indent=4)

# 目标检测并保存检测结果