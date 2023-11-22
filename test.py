from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import models, transforms

import os
from PIL import Image

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

# def read_model():
#     # 读取不同的分类模型
#     dict_class_names = dict()
#     dict_class_names['weather'] = ['clear_weather', 'cloudy', 'foggy', 'rainning', 'snowy']
#     dict_class_names['abnormal_condition'] = ['cracked', 'normal', 'standing_water', 'uneven', 'water']
#     dict_class_names['scerario'] = ['city_road', 'expressway', 'gas_station', 'parking', 'suburbs', 'tunnel']
#     dict_class_names['road_structure'] = ['T-junction', 'crossroad', 'lane_merging', 'normal_road', 'parking-lot', 'ramp', 'round-about']
#     dict_class_names['period'] = ['dawn-or-dusk', 'daytime', 'night', 'unknown']
    
#     dict_class_names['general_obstacle'] = ['speed bumper', 'traffic cone', 'manhole cover', 'water horse', 'stone', 'nothing']
#     # ['bumper', 'cone', 'hole', 'square', 'stone','nothing'] # origin classes
#     dict_class_names['closest_participants_type'] = ['bus', 'passenger car', 'pedestrain', 'policeman', 'truck', 'nothing']
#     dict_class_names['ego_car_behavior'] = ['slow down', 'go straight', 'turn right', 'turn left', 'stop', 'U-turn', 'speed up', 'lane change', 'others']
#     dict_class_names['closest_participants_behavior'] = ['slow down', 'go straight', 'turn right', 'turn left', 'stop', 'U-turn', 'speed up', 'lane change' ,'others']
#     # 在补充完整后面三个类之前不要运行脚本。
    
#     dict_model = dict()
    
#     dict_dir_path = dict()
#     dict_dir_path['period'] = './classification_models/daytime'
#     dict_dir_path['weather'] = './classification_models/weather'
#     dict_dir_path['scerario'] = './classification_models/scerario'
#     dict_dir_path['road_structure'] = './classification_models/roadStructure'
#     dict_dir_path['abnormal_condition'] = './classification_models/abnormal_road'
    
#     dict_dir_path['general_obstacle'] = './detection_models/detection_obstacle'
#     dict_dir_path['closest_participants_type'] = './detection_models/detection_car'
#     dict_dir_path['ego_car_behavior'] = './detection_models/ego_car_behavior'
#     dict_dir_path['closest_participants_behavior'] = './detection_models/closest_participants_behavior'
    
#     model_filename = 'resnet_50_in418_out5_ft.pth'
    
#     yolo_model_filename = 'best.pt'

#     # from a github repo load pytorch base model
#     # repo = 'pytorch/vision'
#     # model_ft = torch.hub.load(repo, 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')

#     # 使用预训练的 ResNet-50 模型
#     model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
#     num_ftrs = model_ft.fc.in_features

#     # 不同的模型有不同的class_names
#     for key, value in dict_class_names.items():
        
#         if key in ['period', 'weather', 'scerario', 'road_structure', 'abnormal_condition']:
#             # classification
#             model_temp = model_ft
#             model_temp.fc = nn.Linear(num_ftrs, len(value))
            
#             dict_model[key] = model_temp.to(device)

#             model_path = os.path.join(dict_dir_path[key], model_filename)
#             print(model_path)
#             dict_model[key].load_state_dict(torch.load(model_path))
#             dict_model[key].eval()
            
#         elif key in ['general_obstacle', 'closest_participants_type']:
#             # object detection and distance estimation
#             model_det = YOLO(os.path.join(dict_dir_path[key], yolo_model_filename))  # load a custom model
#             dict_model[key] = model_det
#         else:
#             continue
            
#     return dict_model, dict_class_names

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

        print('origin_shape is {}'.format(result.orig_shape))
        predicted_class = result.boxes.cls
        confidence = result.boxes.conf
        xyxy = result.boxes.xyxy
        
    # 根据焦距和预先知识计算到相机的距离

    return predicted_class, confidence, xyxy


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

import string

if __name__ == '__main__':
    # find device, cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 加载pytorch resnet50分类模型
    # models, classes = read_model()

    # 加载待检测的MP4文件
    mp4_folder = 'F:\Video\Tianchi_combo/test_video_mp4'
    
    output_folder = './test_video_output'
    os.makedirs(output_folder, exist_ok=True)
    list_mp4 = find_mp4_files(mp4_folder)
    
    data = {'author':'abc', 'time': get_yymmdd(), 'model':'model_name', 'test_results':[]}

    for mp4 in list_mp4:
        index = 1
        mp4_file_path = os.path.join(mp4_folder, mp4)
        print(mp4_file_path)
        
        # 视频名字提取 并修改
        video_name = extract_prefix(mp4)
        if video_name < '60':
            mp4 = video_name + '.avi'
        else:
            mp4 = video_name + '.mp4'
            
        data_piece = {'clip_id': mp4,
            'scerario':'cityroad',
            'weather':'unknown',
            'period':'night',
            'road_structure':'ramp',
            'general_obstacle':'nothing',
            'abnormal_condition':'nothing',
            'ego_car_behavior':'go straight',
            'closest_participants_type':'passenger car',
            'closest_participants_behavior':'go straight'
            }
        
        
        try:
            extract_freq = 4*30 #间隔视频帧, 一个视频取一帧足够了

            cap = cv2.VideoCapture()
            if not cap.open(mp4_file_path):
                print('fail open mp4 file')
                exit(1)
            count = 1
            predicted_class_name = ''
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
                    
                    for key in ['period', 'weather', 'scerario', 'road_structure', 'abnormal_condition', 'closest_participants_type', 'ego_car_behavior', 'closest_participants_behavior']:
                        print('key is {}'.format(key))
                        if key == 'period':
                            cls_one = ['dawn-or-dusk', 'daytime', 'night', 'unknown']
                            # 使用预训练的 ResNet-50 模型
                            model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
                            num_ftrs = model_ft.fc.in_features

                            model_ft.fc = nn.Linear(num_ftrs, len(cls_one))
                            model_first = model_ft
                            model_first = model_first.to(device)
                            folder_first = './classification_models/daytime'
                            model_filename = 'resnet_50_in418_out5_ft.pth'

                            model_path = os.path.join(folder_first, model_filename)
                            # print(model_path)
                            model_first.load_state_dict(torch.load(model_path))
                            model_first.eval()
                        
                            # 预测类别，概率
                            _, predicted_class_name, _ = classification(pil_image, model_first, cls_one, device)
                        
                            if predicted_class_name == 'dawn-or-dusk':
                                predicted_class_name = 'dawn or dusk'
                            # 更新分类结果
                            data_piece[key] = predicted_class_name  
                        if key == 'weather':
                            cls_one = ['clear_weather', 'cloudy', 'foggy', 'rainning', 'snowy']
                            # 使用预训练的 ResNet-50 模型
                            model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
                            num_ftrs = model_ft.fc.in_features

                            model_ft.fc = nn.Linear(num_ftrs, len(cls_one))
                            model_first = model_ft
                            model_first = model_first.to(device)
                            folder_first = './classification_models/weather'
                            model_filename = 'resnet_50_in418_out5_ft.pth'

                            model_path = os.path.join(folder_first, model_filename)
                            # print(model_path)
                            model_first.load_state_dict(torch.load(model_path))
                            model_first.eval()
                        
                            # 预测类别，概率
                            _, predicted_class_name, _ = classification(pil_image, model_first, cls_one, device)
                            # 名词修改
                            if predicted_class_name == 'clear_weather':
                                predicted_class_name = 'clear'
                            # 更新分类结果
                            data_piece[key] = predicted_class_name
                        if key == 'abnormal_condition':
                            cls_one = ['cracked', 'normal', 'standing_water', 'uneven', 'water']
                            # 使用预训练的 ResNet-50 模型
                            model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
                            num_ftrs = model_ft.fc.in_features

                            model_ft.fc = nn.Linear(num_ftrs, len(cls_one))
                            model_first = model_ft
                            model_first = model_first.to(device)
                            folder_first = './classification_models/abnormal_road'
                            model_filename = 'resnet_50_in418_out5_ft.pth'

                            model_path = os.path.join(folder_first, model_filename)
                            # print(model_path)
                            model_first.load_state_dict(torch.load(model_path))
                            model_first.eval()
                        
                            # 预测类别，概率
                            _, predicted_class_name, _ = classification(pil_image, model_first, cls_one, device)
                            
                            if predicted_class_name == 'normal':
                                predicted_class_name = 'nothing'
                            elif predicted_class_name == 'water':
                                predicted_class_name = 'oil or water stain'
                            elif predicted_class_name == 'standing_water':
                                predicted_class_name = 'standing water'
                        
                            # 更新分类结果
                            data_piece[key] = predicted_class_name
                        if key == 'scerario':
                            cls_one = ['city_road', 'expressway', 'gas_station', 'parking', 'suburbs', 'tunnel']
                            # 使用预训练的 ResNet-50 模型
                            model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
                            num_ftrs = model_ft.fc.in_features

                            model_ft.fc = nn.Linear(num_ftrs, len(cls_one))
                            model_first = model_ft
                            model_first = model_first.to(device)
                            folder_first = './classification_models/scerario'
                            model_filename = 'resnet_50_in418_out5_ft.pth'

                            model_path = os.path.join(folder_first, model_filename)
                            # print(model_path)
                            model_first.load_state_dict(torch.load(model_path))
                            model_first.eval()
                        
                            # 预测类别，概率
                            _, predicted_class_name, _ = classification(pil_image, model_first, cls_one, device)
                            
                            if predicted_class_name == 'city_road':
                                predicted_class_name = 'city street'
                            elif predicted_class_name == 'parking':
                                predicted_class_name = 'parking-lot'
                            elif predicted_class_name == 'gas_station':
                                predicted_class_name = 'gas or charging stations'
                        
                            # 更新分类结果
                            data_piece[key] = predicted_class_name
                        if key == 'road_structure':
                            cls_one = ['T-junction', 'crossroad', 'lane_merging', 'normal_road', 'parking-lot', 'ramp', 'round-about']
                            # 使用预训练的 ResNet-50 模型
                            model_ft = models.resnet50(pretrained=False)  # 设为True以加载预训练权重
                            num_ftrs = model_ft.fc.in_features

                            model_ft.fc = nn.Linear(num_ftrs, len(cls_one))
                            model_first = model_ft
                            model_first = model_first.to(device)
                            folder_first = './classification_models/roadStructure'
                            model_filename = 'resnet_50_in418_out5_ft.pth'

                            model_path = os.path.join(folder_first, model_filename)
                            # print(model_path)
                            model_first.load_state_dict(torch.load(model_path))
                            model_first.eval()
                        
                            # 预测类别，概率
                            _, predicted_class_name, _ = classification(pil_image, model_first, cls_one, device)
                            
                            if predicted_class_name == 'crossroad':
                                predicted_class_name = 'crossroads'
                            elif predicted_class_name == 'lane_merging':
                                predicted_class_name = 'lane merging'
                            elif predicted_class_name == 'parking-lot':
                                predicted_class_name = 'parking lot entrance'
                            elif predicted_class_name == 'normal_road':
                                predicted_class_name = 'normal'
                            elif predicted_class_name == 'round-about':
                                predicted_class_name = 'round about'
                            # 更新分类结果
                            data_piece[key] = predicted_class_name
                            
                        # 目标检测
                        if key == 'general_obstacle':
                            cls_one = ['speed bumper', 'traffic cone', 'manhole cover', 'water horse', 'stone', 'nothing']
                            folder_first = './detection_models/detection_obstacle'
                            model_filename = 'best.pt'
                            model_det = YOLO(os.path.join(folder_first, model_filename))
                            # 预测类别
                            predicted_class, confidence, xyxy = object_detect(pil_image, model_det)
                            if predicted_class is not None and len(predicted_class) > 0:
                                predicted_class_name = cls_one[np.argmax(predicted_class.cpu().numpy())] # 取出概率最大的障碍
                                
                                if predicted_class_name == 'bumper':
                                    predicted_class_name = 'speed bumper'
                                elif predicted_class_name == 'hole':
                                    predicted_class_name = 'manhole cover'
                                elif predicted_class_name == 'cone':
                                    predicted_class_name = 'traffic cone'
                                elif predicted_class_name == 'square':
                                    predicted_class_name == 'water horse'
                                
                            else:
                                predicted_class_name = 'nothing'
                            
                            data_piece[key] = predicted_class_name
                        if key == 'closest_participants_type':
                            cls_one = ['bus', 'passenger car', 'pedestrain', 'policeman', 'truck', 'nothing']
                            folder_first = './detection_models/detection_car'
                            model_filename = 'best.pt'
                            model_det = YOLO(os.path.join(folder_first, model_filename))
                            # 预测类别
                            predicted_class, confidence, xyxy = object_detect(pil_image, model_det)
                            if predicted_class is not None and len(predicted_class) > 0:
                                predicted_class_name = cls_one[np.argmax(predicted_class.cpu().numpy())] # 取出概率最大的障碍
                            else:
                                predicted_class_name = 'nothing'
                            data_piece[key] = predicted_class_name
                        if key == 'ego_car_behavior':
                            predicted_class_name = 'go straight'
                            
                            data_piece[key] = predicted_class_name
                        elif key == 'closest_participants_behavior':
                            predicted_class_name = 'go straight'
                            
                            data_piece[key] = predicted_class_name
                            
                    index += 1
                count += 1

            cap.release()
            # 一个视频处理完成，合并data_piece到data中
            data['test_results'].append(data_piece)
            
        except Exception as e:
            print(f"提取帧失败：{mp4_file_path}\n错误信息：{e}")
    # 所有视频处理完成，保存json文件
    with open(os.path.join(output_folder, 'clip_result.json'), 'w') as f:
        json.dump(data, f, indent=4)

# 目标检测并保存检测结果


