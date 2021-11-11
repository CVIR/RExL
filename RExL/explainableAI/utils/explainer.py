import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from tqdm import tqdm_notebook as tqdm

class RLExplainer(nn.Module):
    def __init__(self, model, state_generator, policy, action_space=14, mode='deletion', average_maps=1, all_class=False):
        super(RLExplainer, self).__init__()
        self.model = model
        self.policy = policy
        self.state_generator = state_generator
        self.action_space = action_space
        self.mode = mode
        self.scale = int(224/self.action_space)
        self.softmax = nn.Softmax(dim=1)
        self.average_maps = average_maps
        self.all_class = all_class

    def forward(self, x_):
        N, _, H, W = x_.shape
        
        x_rand_ = (torch.rand_like(self.currentImage)-0.5)*255

        saliency = None
        saliency_final = None
        rewards_old = None
        class_indices = None
        predictions = None
        rewards = None

        if not self.all_class:
            predictions = self.softmax(self.model(x_))
            rewards_old = np.zeros((N))
            class_indices = torch.argmax(predictions, dim=1).cpu().numpy()          
             
            saliency = np.zeros((N, self.action_space, self.action_space))
            saliency_final = np.zeros((N, H, W))
        else:     
            predictions = self.softmax(self.model(x_)).cpu().numpy()
            N, numClass = predictions.shape
            rewards_old = np.zeros((N, numClass))
            
            saliency = np.zeros((N, numClass, self.action_space, self.action_space))
            saliency_final = np.zeros((N, numClass, H, W))
            
        N, numClass = predictions.shape

        for i in tqdm(range(self.average_maps), desc='Average:'):
            
            rewards_old = np.zeros_like(rewards_old)

            if self.mode == 'insertion':
                x = x_.clone().detach()
                x_canvas = x_rand_.clone().detach()
            else:
                x_canvas = x_.clone().detach()
                x = x_rand_.clone().detach()
                predictions = self.softmax(self.model(x_)).cpu().numpy()
                rewards_old = predictions[:, class_indices]
                
            states, _ = self.state_generator.getState(x, x_canvas)
            for j in tqdm(range(self.action_space*self.action_space), desc="Steps:"):
                actions, _, _, _ = self.policy.step(states.cpu())
                for i in range(N):
                    x_canvas[i, :, actions[i, 0]*self.scale:(actions[i, 0]+1)*self.scale, actions[i, 1]*self.scale:(actions[i, 1]+1)*self.scale] = x[i, :, actions[i, 0]*self.scale:(actions[i, 0]+1)*self.scale, actions[i, 1]*self.scale:(actions[i, 1]+1)*self.scale]

                predictions = self.softmax(self.model(x_canvas)).cpu().numpy()
                if not self.all_class:
                    rewards = predictions[np.arange(N), class_indices]
                else:
                    rewards = predictions

                if not self.all_class:
                    saliency[np.arange(N), actions[:, 0], actions[:, 1]] += (rewards - rewards_old) * ( 1 if self.mode == 'insertion' else -1 )
                else:
                    saliency[np.arange(N), :, actions[:, 0], actions[:, 1]] += (rewards - rewards_old) * ( 1 if self.mode == 'insertion' else -1 )
                rewards_old = rewards
                states, _ = self.state_generator.getState(x, x_canvas)
        
        self.state_generator.clear_cache()
        saliency /= self.average_maps
        
        if not self.all_class:
            for i in range(N):
                sigma = np.min(saliency[i])
                saliency[i] = ( -sigma if (sigma < 0 ) else  0 ) + saliency[i]
                saliency_final[i] = resize(saliency[i], (H,W), order=1, mode='reflect',anti_aliasing=False)
        else :
            for i in range(N):
                for j in range(numClass):
                    sigma = np.min(saliency[i, j])
                    saliency[i, j] = ( -sigma if (sigma < 0 ) else  0 ) + saliency[i, j]
                    saliency_final[i, j] = resize(saliency[i, j], (H,W), order=1, mode='reflect',anti_aliasing=False)
        
        return saliency_final