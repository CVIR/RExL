import gym
from gym import spaces

# from RISE.evaluation import gkern

from explainableAI.utils import getImageNet, getPASCAL, getMSCOCO, getPASCAL12
from explainableAI.utils import load_vgg, load_resnet, load_effnet
from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch

import nvgpu
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


class RLStateGenerator:
    def __init__(self, model, no_of_class = 1, training_type='deletion', device=torch.device('cpu')):
        self.model = model
        self.features = model.features
        self.no_of_class = no_of_class
        self.type = training_type
        self.cache_hot = False
        self.device = device

    def clear_cache(self):
        self.cache_hot = False

    def getState(self, x, x_canvas, class_id = None):
        ft_canvas = None
        
        if self.type == 'insertion':      
            
            if not self.cache_hot:
                self.current_image_feature_cache = self.features(x)          
            ft_canvas = self.features(x_canvas)

            if self.no_of_class > 1:               

                n, c, h, w = self.current_image_feature_cache.shape
                              
                if not self.cache_hot:
                    one_hot = torch.zeros(n, self.no_of_class, h, w).type(torch.FloatTensor).to(self.device)
                    if class_id != None:
                        one_hot[:, class_id, :, :] = 1
                        self.one_hot_cache = one_hot
                        self.cache_hot = True
                    else :
                        pred = self.model(x)
                        self.predicted_class_cache = torch.argmax(pred, dim=1)
                        one_hot[0:n, self.predicted_class_cache,:,:] = 1
                        self.one_hot_cache = one_hot
                        self.cache_hot = True
                
                state = torch.cat((self.one_hot_cache, ft_canvas, self.current_image_feature_cache), dim=1) 
            else :
                state = torch.cat((ft_canvas, self.current_image_feature_cache), dim=1)
        else :
            ft_canvas = self.features(x_canvas)                
                
            if self.no_of_class > 1:
                n, c, h, w = ft_canvas.shape

                if not self.cache_hot:
                    one_hot = torch.zeros(n, self.no_of_class, h, w).type(torch.FloatTensor).to(self.device)  
                    if class_id != None:
                        one_hot[:, class_id, :, :] = 1
                        self.one_hot_cache = one_hot
                        self.cache_hot = True
                    else :
                        pred = self.model(x_canvas)
                        self.predicted_class_cache = torch.argmax(pred, dim=1)
                        one_hot[0:n, self.predicted_class_cache,:,:] = 1
                        self.one_hot_cache = one_hot
                        self.cache_hot = True

                state = torch.cat((self.one_hot_cache, ft_canvas), dim=1)
            else :
                state = ft_canvas

        return state, ft_canvas


class AttentionEnv(gym.Env):
    def __init__(self, model_name='vgg', kernelSize = 14, refresh_time = 1, class_index = -1, no_of_training_classes=None, RootPath=None, datasetName='PASCAL', datasetType = 'train', mode_of_training = 'deletion', batch_size=1, vid_path=None, idx = -1, device = None):
        
        print("Running on class ", class_index)
        print("Running on idx ", idx)
        self.max_steps = 49
        self.datasetName = datasetName
        self.refresh_time = refresh_time
        self.model_name = model_name
        self.idx = idx

        if RootPath is None:
            sys.exit("RootPath is set to None.")

        if class_index != -1 and no_of_training_classes is None and datasetName == 'IMAGENET':
            sys.exit("no_of_training_classes need to be passed when using class_index for ImageNet")

        if no_of_training_classes is None:
            if datasetName == 'MSCOCO':
                no_of_training_classes = 80
            elif datasetName=='PASCAL' or datasetName == 'PASCAL12':
                no_of_training_classes = 20
            elif datasetName=='IMAGENET':
                no_of_training_classes = 1000
        
        if model_name == 'vgg':
            self.model, self.device = load_vgg(RootPath, datasetName, device=device)
            
        elif model_name == 'resnet':
            self.model, self.device = load_resnet(RootPath, datasetName, device=device)

        elif model_name == 'effnet':
            self.model, self.device = load_effnet(RootPath, datasetName, device=device)

        self.backbone, self.backbone_device = load_resnet(RootPath, datasetName, device=device)

        self.no_of_training_classes = no_of_training_classes
        self.dataset = None
        self.class_index = class_index
        if datasetName == 'MSCOCO':
            self.dataset = getMSCOCO(RootPath, datasetType, class_index, idx=idx)    
        elif datasetName == 'PASCAL':
            self.dataset = getPASCAL(RootPath, datasetType, class_index, idx=idx)
        elif datasetName == 'PASCAL12':
            self.dataset = getPASCAL12(RootPath, datasetType, class_index, idx=idx)
        elif datasetName == 'IMAGENET':
            self.dataset = getImageNet(RootPath, datasetType, self.class_index, no_of_training_classes=1, idx = idx)

        self.class_index = class_index
        shuffle = datasetType=='train'
        if batch_size < 8:
            self.dataLoader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        else : 
            self.dataLoader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        
        self.dataIterator = iter(self.dataLoader)        

        self.plotter = None
        self.kernelSize = kernelSize
        self.mode_of_training = mode_of_training
        self.softmax = nn.Softmax(dim=1)
        
        self.episodeNumber = 0
        self.exhaustedClasses = 1
        self.totalClasses = 1

        self.state_generator = RLStateGenerator(self.backbone, training_type = self.mode_of_training, no_of_class=no_of_training_classes, device=self.backbone_device)

        self.action_space = spaces.Discrete(kernelSize*kernelSize)

        if mode_of_training == 'insertion':
            if class_index == -1:
                self.observation_space = spaces.Box(low=0, high=float('Inf'), shape=(4096+no_of_training_classes, 7, 7), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0, high=float('Inf'), shape=(4096, 7, 7), dtype=np.float32)
        else :
            if class_index == -1:
                self.observation_space = spaces.Box(low=0, high=float('Inf'), shape=(2048+no_of_training_classes, 7, 7), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0, high=float('Inf'), shape=(2048, 7, 7), dtype=np.float32)

        self.counter=0
        self.vid_path = vid_path
        self.current_class = self.class_index

    def reset(self, sample=None, class_id = -1):
        self.i = 0
        if sample is None:
            try:
                dataDict = next(self.dataIterator)
            except StopIteration:                
                self.dataIterator = iter(self.dataLoader)
                dataDict = next(self.dataIterator)

        else:
            dataDict = sample

        self.episodeNumber += 1
        self.step_count = 0

        self.currentImage = dataDict['image'].type(torch.FloatTensor).to(self.device)
        # self.bboxes = dataDict['bboxes']
        self.img_path = dataDict['image_path']
        print(dataDict['image_path'])
        self.canvas_visited = np.zeros((self.kernelSize,self.kernelSize))
        pred = self.softmax(self.model(self.currentImage))
        if class_id == -1:
            if self.class_index == -1:
                self.current_class = torch.argmax(pred, dim=1).cpu().numpy()[0]
            else:
                self.current_class = self.class_index
        else:
            self.current_class = class_id
        
        self.max_pred = torch.max(pred).cpu().numpy()
        if self.mode_of_training == 'insertion':
            if self.datasetName != 'IMAGENET' and self.datasetName != 'PASCAL12':
                self.canvas = (torch.rand_like(self.currentImage)-0.5)*255
            else:
                self.canvas = (torch.rand_like(self.currentImage)-0.5)
        else: 
            self.canvas = self.currentImage.clone().detach()
            if self.datasetName != 'IMAGENET' and self.datasetName != 'PASCAL12':
                self.currentImage = (torch.rand_like(self.canvas) -0.5)*255
            else:
                self.currentImage = (torch.rand_like(self.canvas) -0.5)

        print(self.current_class)
        self.state_generator.clear_cache()  
        self.rewards=[]
        obs, _ = self.state_generator.getState(self.currentImage, self.canvas, class_id = self.current_class) 
        return obs[0].cpu()

    def step(self, action):
        self.i += 1
        self.counter += 1
        x = action // self.kernelSize
        y = action % self.kernelSize
        total_reward = 0
        done = False
        
        self.step_count += 1

        pixels = int(224/self.kernelSize)
        
        if self.canvas_visited[x, y] == False:
            self.canvas[0, :, x*pixels:(x+1)*pixels, y*pixels:(y+1)*pixels] = self.currentImage[0, :, x*pixels:(x+1)*pixels, y*pixels:(y+1)*pixels]
            self.canvas_visited[x, y] = True
     
        obs, canvas_ft = self.state_generator.getState(self.currentImage, self.canvas, class_id=self.current_class)
        predProbs = self.softmax(self.model(self.canvas))

        top_pred = torch.argmax(predProbs, dim=1).cpu().numpy()[0]

        if self.step_count == self.max_steps:
            done = True
                
        total_reward = predProbs[0, self.current_class]

        self.rewards.append(total_reward.cpu().item())
        
        if self.mode_of_training == 'deletion':
            total_reward = -total_reward

        if self.vid_path is not None:
            print(self.vid_path)
            self.render(save_to=self.vid_path)

        
        return obs[0].cpu(), total_reward.cpu().sum().item(), done, {}
        
    def seed(self, value):
        np.random.seed(value)
        torch.manual_seed(value)

    def render(self,mode='rgb_array', save_to=None):
        tempImage = self.canvas[0].cpu().numpy().copy()
        tempImage1 = self.currentImage[0].cpu().numpy().copy()
        # denormalize to visualize
        
        if self.datasetName == 'IMAGENET' or self.datasetName == 'PASCAL12':
            mean = [0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            
            for channel in range(3):
                tempImage[channel] = tempImage[channel]*std[channel] + mean[channel]
                tempImage1[channel] = tempImage1[channel]*std[channel] + mean[channel]
                
            tempImage = tempImage*255
            tempImage1 = tempImage1*255
            tempImage = tempImage[(2,1,0),:,:]
            tempImage1 = tempImage1[(2,1,0),:,:]
            
        else:
            mean = [104.01, 116.67, 122.68]
            for channel in range(3):
                tempImage[channel] = tempImage[channel] + mean[channel]
                tempImage1[channel] = tempImage1[channel] + mean[channel]
        # Uncomment if matplotlib is used
        # tempImage = tempImage[(2,1,0), :, :]

        tempImage = tempImage.transpose((1,2,0))
        tempImage1 = tempImage1.transpose((1,2,0))

        # Visualize the Image
        tempImage[tempImage < 0] = 0
        tempImage1[tempImage1 < 0] = 0


        tempImage1 = tempImage1.astype(np.uint8)
        tempImage = tempImage.astype(np.uint8)
        cv2.imwrite(os.path.join(save_to, '%06d'%self.counter + '.jpg'), tempImage)

    def close(self):
        # close window for showing images and quit
        self.model.cpu()
