import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
import sys
import os
import nvgpu

class VGG_FCN(nn.Module):
    def __init__(self, model_ft):
        super(VGG_FCN, self).__init__()
        self.features = model_ft.features
        self.classifier = model_ft.classifier

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x[:,:,0,0]
        return x
    
class ResNet_Imgnet(nn.Module):
    def __init__(self, model):
        super(ResNet_Imgnet, self).__init__()
        self.modules = list(model.children())
        self.features = nn.Sequential(*(self.modules[:-2]))
        self.adapt_pool = nn.Sequential(self.modules[-2])
        self.fc = nn.Sequential(self.modules[-1])

    def classifier(self, X):
        X = self.adapt_pool(X)
        X = X[:, :, 0, 0]
        X = self.fc(X)
        return X
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
class Resnet50(nn.Module):
    def __init__(self, numClass):
        super(Resnet50, self).__init__()
        self.Base= torch.nn.Sequential()
        self.Base.add_module("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=True))
        self.Base.add_module("bn_conv1", nn.BatchNorm2d(64))
        self.Base.add_module("conv1_relu", nn.ReLU())
        self.Base.add_module("pool1", nn.MaxPool2d(3, 2))
        
        self.Res2a_b1= torch.nn.Sequential()
        self.Res2a_b1.add_module("res2a_branch1", nn.Conv2d(64, 256, 1, stride=1, padding=0, bias=False))
        self.Res2a_b1.add_module("scale2a_branch1", nn.BatchNorm2d(256))
        
        self.Res2a_b2= torch.nn.Sequential()
        self.Res2a_b2.add_module("res2a_branch2a", nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False))
        self.Res2a_b2.add_module("scale2a_branch2a", nn.BatchNorm2d(64))
        self.Res2a_b2.add_module("res2a_branch2a_relu", nn.ReLU())
        self.Res2a_b2.add_module("res2a_branch2b", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
        self.Res2a_b2.add_module("scale2a_branch2b", nn.BatchNorm2d(64))
        self.Res2a_b2.add_module("res2a_branch2b_relu", nn.ReLU())
        self.Res2a_b2.add_module("res2a_branch2c", nn.Conv2d(64, 256, 1, stride=1, padding=0, bias=False))
        self.Res2a_b2.add_module("scale2a_branch2c", nn.BatchNorm2d(256))
        
        self.Res2b_b2= torch.nn.Sequential()
        self.Res2b_b2.add_module("res2b_branch2a", nn.Conv2d(256, 64, 1, stride=1, padding=0, bias=False))
        self.Res2b_b2.add_module("scale2b_branch2a", nn.BatchNorm2d(64))
        self.Res2b_b2.add_module("res2b_branch2a_relu", nn.ReLU())
        self.Res2b_b2.add_module("res2b_branch2b", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
        self.Res2b_b2.add_module("scale2b_branch2b", nn.BatchNorm2d(64))
        self.Res2b_b2.add_module("res2b_branch2b_relu", nn.ReLU())
        self.Res2b_b2.add_module("res2b_branch2c", nn.Conv2d(64, 256, 1, stride=1, padding=0, bias=False))
        self.Res2b_b2.add_module("scale2b_branch2c", nn.BatchNorm2d(256))
        
        self.Res2c_b2= torch.nn.Sequential()
        self.Res2c_b2.add_module("res2c_branch2a", nn.Conv2d(256, 64, 1, stride=1, padding=0, bias=False))
        self.Res2c_b2.add_module("scale2c_branch2a", nn.BatchNorm2d(64))
        self.Res2c_b2.add_module("res2c_branch2a_relu", nn.ReLU())
        self.Res2c_b2.add_module("res2c_branch2b", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False))
        self.Res2c_b2.add_module("scale2c_branch2b", nn.BatchNorm2d(64))
        self.Res2c_b2.add_module("res2c_branch2b_relu", nn.ReLU())
        self.Res2c_b2.add_module("res2c_branch2c", nn.Conv2d(64, 256, 1, stride=1, padding=0, bias=False))
        self.Res2c_b2.add_module("scale2c_branch2c", nn.BatchNorm2d(256))
        
        # =================================================================
        self.Res3a_b1= torch.nn.Sequential()
        self.Res3a_b1.add_module("res3a_branch1", nn.Conv2d(256, 512, 1, stride=2, padding=0, bias=False))
        self.Res3a_b1.add_module("scale3a_branch1", nn.BatchNorm2d(512))
        
        self.Res3a_b2= torch.nn.Sequential()
        self.Res3a_b2.add_module("res3a_branch2a", nn.Conv2d(256, 128, 1, stride=2, padding=0, bias=False))
        self.Res3a_b2.add_module("scale3a_branch2a", nn.BatchNorm2d(128))
        self.Res3a_b2.add_module("res3a_branch2a_relu", nn.ReLU())
        self.Res3a_b2.add_module("res3a_branch2b", nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False))
        self.Res3a_b2.add_module("scale3a_branch2b", nn.BatchNorm2d(128))
        self.Res3a_b2.add_module("res3a_branch2b_relu", nn.ReLU())
        self.Res3a_b2.add_module("res3a_branch2c", nn.Conv2d(128, 512, 1, stride=1, padding=0, bias=False))
        self.Res3a_b2.add_module("scale3a_branch2c", nn.BatchNorm2d(512))
        
        self.Res3b_b2= torch.nn.Sequential()
        self.Res3b_b2.add_module("res3b_branch2a", nn.Conv2d(512, 128, 1, stride=1, padding=0, bias=False))
        self.Res3b_b2.add_module("scale3b_branch2a", nn.BatchNorm2d(128))
        self.Res3b_b2.add_module("res3b_branch2a_relu", nn.ReLU())
        self.Res3b_b2.add_module("res3b_branch2b", nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False))
        self.Res3b_b2.add_module("scale3b_branch2b", nn.BatchNorm2d(128))
        self.Res3b_b2.add_module("res3b_branch2b_relu", nn.ReLU())
        self.Res3b_b2.add_module("res3b_branch2c", nn.Conv2d(128, 512, 1, stride=1, padding=0, bias=False))
        self.Res3b_b2.add_module("scale3b_branch2c", nn.BatchNorm2d(512))
        
        self.Res3c_b2= torch.nn.Sequential()
        self.Res3c_b2.add_module("res3c_branch2a", nn.Conv2d(512, 128, 1, stride=1, padding=0, bias=False))
        self.Res3c_b2.add_module("scale3c_branch2a", nn.BatchNorm2d(128))
        self.Res3c_b2.add_module("res3c_branch2a_relu", nn.ReLU())
        self.Res3c_b2.add_module("res3c_branch2b", nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False))
        self.Res3c_b2.add_module("scale3c_branch2b", nn.BatchNorm2d(128))
        self.Res3c_b2.add_module("res3c_branch2b_relu", nn.ReLU())
        self.Res3c_b2.add_module("res3c_branch2c", nn.Conv2d(128, 512, 1, stride=1, padding=0, bias=False))
        self.Res3c_b2.add_module("scale3c_branch2c", nn.BatchNorm2d(512))
        
        self.Res3d_b2= torch.nn.Sequential()
        self.Res3d_b2.add_module("res3d_branch2a", nn.Conv2d(512, 128, 1, stride=1, padding=0, bias=False))
        self.Res3d_b2.add_module("scale3d_branch2a", nn.BatchNorm2d(128))
        self.Res3d_b2.add_module("res3d_branch2a_relu", nn.ReLU())
        self.Res3d_b2.add_module("res3d_branch2b", nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False))
        self.Res3d_b2.add_module("scale3d_branch2b", nn.BatchNorm2d(128))
        self.Res3d_b2.add_module("res3d_branch2b_relu", nn.ReLU())
        self.Res3d_b2.add_module("res3d_branch2c", nn.Conv2d(128, 512, 1, stride=1, padding=0, bias=False))
        self.Res3d_b2.add_module("scale3d_branch2c", nn.BatchNorm2d(512))
        
        # =================================================================
        self.Res4a_b1= torch.nn.Sequential()
        self.Res4a_b1.add_module("res4a_branch1", nn.Conv2d(512, 1024, 1, stride=2, padding=0, bias=False))
        self.Res4a_b1.add_module("scale4a_branch1", nn.BatchNorm2d(1024))
        
        self.Res4a_b2= torch.nn.Sequential()
        self.Res4a_b2.add_module("res4a_branch2a", nn.Conv2d(512, 256, 1, stride=2, padding=0, bias=False))
        self.Res4a_b2.add_module("scale4a_branch2a", nn.BatchNorm2d(256))
        self.Res4a_b2.add_module("res4a_branch2a_relu", nn.ReLU())
        self.Res4a_b2.add_module("res4a_branch2b", nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False))
        self.Res4a_b2.add_module("scale4a_branch2b", nn.BatchNorm2d(256))
        self.Res4a_b2.add_module("res4a_branch2b_relu", nn.ReLU())
        self.Res4a_b2.add_module("res4a_branch2c", nn.Conv2d(256, 1024, 1, stride=1, padding=0, bias=False))
        self.Res4a_b2.add_module("scale4a_branch2c", nn.BatchNorm2d(1024))
        
        self.Res4b_b2= torch.nn.Sequential()
        self.Res4b_b2.add_module("res4b_branch2a", nn.Conv2d(1024, 256, 1, stride=1, padding=0, bias=False))
        self.Res4b_b2.add_module("scale4b_branch2a", nn.BatchNorm2d(256))
        self.Res4b_b2.add_module("res4b_branch2a_relu", nn.ReLU())
        self.Res4b_b2.add_module("res4b_branch2b", nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False))
        self.Res4b_b2.add_module("scale4b_branch2b", nn.BatchNorm2d(256))
        self.Res4b_b2.add_module("res4b_branch2b_relu", nn.ReLU())
        self.Res4b_b2.add_module("res4b_branch2c", nn.Conv2d(256, 1024, 1, stride=1, padding=0, bias=False))
        self.Res4b_b2.add_module("scale4b_branch2c", nn.BatchNorm2d(1024))
        
        self.Res4c_b2= torch.nn.Sequential()
        self.Res4c_b2.add_module("res4c_branch2a", nn.Conv2d(1024, 256, 1, stride=1, padding=0, bias=False))
        self.Res4c_b2.add_module("scale4c_branch2a", nn.BatchNorm2d(256))
        self.Res4c_b2.add_module("res4c_branch2a_relu", nn.ReLU())
        self.Res4c_b2.add_module("res4c_branch2b", nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False))
        self.Res4c_b2.add_module("scale4c_branch2b", nn.BatchNorm2d(256))
        self.Res4c_b2.add_module("res4c_branch2b_relu", nn.ReLU())
        self.Res4c_b2.add_module("res4c_branch2c", nn.Conv2d(256, 1024, 1, stride=1, padding=0, bias=False))
        self.Res4c_b2.add_module("scale4c_branch2c", nn.BatchNorm2d(1024))
        
        self.Res4d_b2= torch.nn.Sequential()
        self.Res4d_b2.add_module("res4d_branch2a", nn.Conv2d(1024, 256, 1, stride=1, padding=0, bias=False))
        self.Res4d_b2.add_module("scale4d_branch2a", nn.BatchNorm2d(256))
        self.Res4d_b2.add_module("res4d_branch2a_relu", nn.ReLU())
        self.Res4d_b2.add_module("res4d_branch2b", nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False))
        self.Res4d_b2.add_module("scale4d_branch2b", nn.BatchNorm2d(256))
        self.Res4d_b2.add_module("res4d_branch2b_relu", nn.ReLU())
        self.Res4d_b2.add_module("res4d_branch2c", nn.Conv2d(256, 1024, 1, stride=1, padding=0, bias=False))
        self.Res4d_b2.add_module("scale4d_branch2c", nn.BatchNorm2d(1024))
        
        self.Res4e_b2= torch.nn.Sequential()
        self.Res4e_b2.add_module("res4e_branch2a", nn.Conv2d(1024, 256, 1, stride=1, padding=0, bias=False))
        self.Res4e_b2.add_module("scale4e_branch2a", nn.BatchNorm2d(256))
        self.Res4e_b2.add_module("res4e_branch2a_relu", nn.ReLU())
        self.Res4e_b2.add_module("res4e_branch2b", nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False))
        self.Res4e_b2.add_module("scale4e_branch2b", nn.BatchNorm2d(256))
        self.Res4e_b2.add_module("res4e_branch2b_relu", nn.ReLU())
        self.Res4e_b2.add_module("res4e_branch2c", nn.Conv2d(256, 1024, 1, stride=1, padding=0, bias=False))
        self.Res4e_b2.add_module("scale4e_branch2c", nn.BatchNorm2d(1024))
        
        self.Res4f_b2= torch.nn.Sequential()
        self.Res4f_b2.add_module("res4f_branch2a", nn.Conv2d(1024, 256, 1, stride=1, padding=0, bias=False))
        self.Res4f_b2.add_module("scale4f_branch2a", nn.BatchNorm2d(256))
        self.Res4f_b2.add_module("res4f_branch2a_relu", nn.ReLU())
        self.Res4f_b2.add_module("res4f_branch2b", nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False))
        self.Res4f_b2.add_module("scale4f_branch2b", nn.BatchNorm2d(256))
        self.Res4f_b2.add_module("res4f_branch2b_relu", nn.ReLU())
        self.Res4f_b2.add_module("res4f_branch2c", nn.Conv2d(256, 1024, 1, stride=1, padding=0, bias=False))
        self.Res4f_b2.add_module("scale4f_branch2c", nn.BatchNorm2d(1024))
        
        # =================================================================
        self.Res5a_b1= torch.nn.Sequential()
        self.Res5a_b1.add_module("res5a_branch1", nn.Conv2d(1024, 2048, 1, stride=2, padding=0, bias=False))
        self.Res5a_b1.add_module("scale5a_branch1", nn.BatchNorm2d(2048))
        
        self.Res5a_b2= torch.nn.Sequential()
        self.Res5a_b2.add_module("res5a_branch2a", nn.Conv2d(1024, 512, 1, stride=2, padding=0, bias=False))
        self.Res5a_b2.add_module("scale5a_branch2a", nn.BatchNorm2d(512))
        self.Res5a_b2.add_module("res5a_branch2a_relu", nn.ReLU())
        self.Res5a_b2.add_module("res5a_branch2b", nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False))
        self.Res5a_b2.add_module("scale5a_branch2b", nn.BatchNorm2d(512))
        self.Res5a_b2.add_module("res5a_branch2b_relu", nn.ReLU())
        self.Res5a_b2.add_module("res5a_branch2c", nn.Conv2d(512, 2048, 1, stride=1, padding=0, bias=False))
        self.Res5a_b2.add_module("scale5a_branch2c", nn.BatchNorm2d(2048))
        
        self.Res5b_b2= torch.nn.Sequential()
        self.Res5b_b2.add_module("res5b_branch2a", nn.Conv2d(2048, 512, 1, stride=1, padding=0, bias=False))
        self.Res5b_b2.add_module("scale5b_branch2a", nn.BatchNorm2d(512))
        self.Res5b_b2.add_module("res5b_branch2a_relu", nn.ReLU())
        self.Res5b_b2.add_module("res5b_branch2b", nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False))
        self.Res5b_b2.add_module("scale5b_branch2b", nn.BatchNorm2d(512))
        self.Res5b_b2.add_module("res5b_branch2b_relu", nn.ReLU())
        self.Res5b_b2.add_module("res5b_branch2c", nn.Conv2d(512, 2048, 1, stride=1, padding=0, bias=False))
        self.Res5b_b2.add_module("scale5b_branch2c", nn.BatchNorm2d(2048))
        
        self.Res5c_b2= torch.nn.Sequential()
        self.Res5c_b2.add_module("res5c_branch2a", nn.Conv2d(2048, 512, 1, stride=1, padding=0, bias=False))
        self.Res5c_b2.add_module("scale5c_branch2a", nn.BatchNorm2d(512))
        self.Res5c_b2.add_module("res5c_branch2a_relu", nn.ReLU())
        self.Res5c_b2.add_module("res5c_branch2b", nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False))
        self.Res5c_b2.add_module("scale5c_branch2b", nn.BatchNorm2d(512))
        self.Res5c_b2.add_module("res5c_branch2b_relu", nn.ReLU())
        self.Res5c_b2.add_module("res5c_branch2c", nn.Conv2d(512, 2048, 1, stride=1, padding=0, bias=False))
        self.Res5c_b2.add_module("scale5c_branch2c", nn.BatchNorm2d(2048))
        
        # =================================================================
        self.pool5= torch.nn.Sequential()
        self.pool5.add_module("pool5", nn.AvgPool2d(7))
        
        # =================================================================
        self.fc8_20_conv= torch.nn.Sequential()
        self.fc8_20_conv.add_module("fc8_20_conv", nn.Conv2d(2048, numClass, 1, stride=1, padding=0, bias=False))
        
    def features(self, x):
        x = self.Base.forward(x)
        Res2a_b1 = self.Res2a_b1.forward(x)
        Res2a_b2 = self.Res2a_b2.forward(x)
        res2a = nn.ReLU()(Res2a_b1 + Res2a_b2)
        res2b_b2c = self.Res2b_b2.forward(res2a)
        res2b = nn.ReLU()(res2a + res2b_b2c)
        res2c_b2c = self.Res2c_b2.forward(res2b)
        res2c = nn.ReLU()(res2b + res2c_b2c)
        
        Res3a_b1 = self.Res3a_b1.forward(res2c)
        Res3a_b2 = self.Res3a_b2.forward(res2c)
        res3a = nn.ReLU()(Res3a_b1 + Res3a_b2)
        res3b_b2c = self.Res3b_b2.forward(res3a)
        res3b = nn.ReLU()(res3a + res3b_b2c)
        res3c_b2c = self.Res3c_b2.forward(res3b)
        res3c = nn.ReLU()(res3b + res3c_b2c)
        res3d_b2c = self.Res3d_b2.forward(res3c)
        res3d = nn.ReLU()(res3c + res3d_b2c)
        
        Res4a_b1 = self.Res4a_b1.forward(res3d)
        Res4a_b2 = self.Res4a_b2.forward(res3d)
        res4a = nn.ReLU()(Res4a_b1 + Res4a_b2)
        res4b_b2c = self.Res4b_b2.forward(res4a)
        res4b = nn.ReLU()(res4a + res4b_b2c)
        res4c_b2c = self.Res4c_b2.forward(res4b)
        res4c = nn.ReLU()(res4b + res4c_b2c)
        res4d_b2c = self.Res4d_b2.forward(res4c)
        res4d = nn.ReLU()(res4c + res4d_b2c)
        res4e_b2c = self.Res4e_b2.forward(res4d)
        res4e = nn.ReLU()(res4d + res4e_b2c)
        res4f_b2c = self.Res4f_b2.forward(res4e)
        res4f = nn.ReLU()(res4e + res4f_b2c)
        
        Res5a_b1 = self.Res5a_b1.forward(res4f)
        Res5a_b2 = self.Res5a_b2.forward(res4f)
        res5a = nn.ReLU()(Res5a_b1 + Res5a_b2)
        res5b_b2c = self.Res5b_b2.forward(res5a)
        res5b = nn.ReLU()(res5a + res5b_b2c)
        res5c_b2c = self.Res5c_b2.forward(res5b)
        res5c = nn.ReLU()(res5b + res5c_b2c)
        return res5c
        
    def classifier(self, x):
        pool5 = self.pool5.forward(x)
        fc8 = self.fc8_20_conv.forward(pool5)
        return fc8[:,:,0,0]
        
    def forward(self, x):
        ftrs = self.features(x)
        res = self.classifier(ftrs)
        return res

def load_vgg(RootPath, datasetName='PASCAL', device=None):

    use_gpu = torch.cuda.is_available()
    if device is None:
        gpu_info = nvgpu.gpu_info()  
        compute_device = torch.device("cpu")
        for i in range(len(gpu_info)):
            print("MemTotal: ", gpu_info[i]["mem_total"], "MemUsed: ", gpu_info[i]["mem_used"])
            if gpu_info[i]["mem_total"] - gpu_info[i]["mem_used"] > 8000:
                compute_device = torch.device("cuda:"+str(i))
                break
    else:
        compute_device = device

    print("Loading model to compute device", compute_device)
    
    numClass = 0
    if datasetName == 'MSCOCO':
        numClass = 80
    elif datasetName=='PASCAL':
        numClass = 20
    
    preTrainedModel = None
    if datasetName == 'MSCOCO':
        preTrainedModel = RootPath+'FT_Models/Caffe_model/vgg16_mscoco.pth.tar'
    elif datasetName == 'PASCAL':
        preTrainedModel = RootPath+'Models/vgg16_pascal07_.pth.tar'

    model_ft = models.vgg16(pretrained=True)
    
    if datasetName != 'IMAGENET':
        model_ft.classifier = nn.Sequential(
                        nn.Conv2d(512, 4096, 7),
                        nn.ReLU(),
                        nn.Conv2d(4096, 4096, 1),
                        nn.ReLU(),
                        nn.Conv2d(4096, numClass, 1)
        )
        model_ft = VGG_FCN(model_ft)

        if not preTrainedModel is None and os.path.isfile(preTrainedModel):
            print("=> loading state_dict '{}'".format(preTrainedModel))
            state_dict = torch.load(preTrainedModel)
            model_ft.load_state_dict(state_dict)
            print("=> loaded state_dict '{}'".format(preTrainedModel))
        else:
            sys.exit("state_dict not found. Cannot load pretrained model.")
    
    for p in model_ft.parameters():
        p.requires_grad = False

    model_ft.eval()
    
    arch = model_ft.__class__.__name__
    print(arch)

    if use_gpu:
        model_ft = model_ft.to(compute_device)

    return model_ft, compute_device

def load_resnet(RootPath, datasetName='IMAGENET', device = None):

    use_gpu = torch.cuda.is_available()
    if device is None:
        
        
        gpu_info = nvgpu.gpu_info()  
        compute_device = torch.device("cpu")
        for i in range(len(gpu_info)):
            print("MemTotal: ", gpu_info[i]["mem_total"], "MemUsed: ", gpu_info[i]["mem_used"])
            if gpu_info[i]["mem_total"] - gpu_info[i]["mem_used"] > 3000:
                compute_device = torch.device("cuda:"+str(i))
                break
    else:
        compute_device = device

    print("Loading model to compute device", compute_device)
    
    if datasetName=='IMAGENET' or datasetName == 'PASCAL12':
        model_ft = models.resnet50(pretrained=True)
        if datasetName == 'PASCAL12':
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 20)
    
        if datasetName == 'PASCAL12':
            preTrainedModel = RootPath + 'Models/resnet50_pascal12_2.pth.tar'
            if not preTrainedModel is None and os.path.isfile(preTrainedModel):
                print("=> loading state_dict '{}'".format(preTrainedModel))
                state_dict = torch.load(preTrainedModel)
                model_ft.load_state_dict(state_dict['state_dict'])
                print("=> loaded state_dict '{}'".format(preTrainedModel))
            else:
                sys.exit("state_dict not found. Cannot load pretrained model.")

        model_ft = ResNet_Imgnet(model_ft)        
        
    else:
        if datasetName == 'MSCOCO':
            numClass = 80
        elif datasetName=='PASCAL':
            numClass = 20
            
        model_ft = Resnet50(numClass)
        if datasetName == 'PASCAL':
            preTrainedModel = RootPath + 'Models/ResNet50_voc.pth.tar'
            if not preTrainedModel is None and os.path.isfile(preTrainedModel):
                print("=> loading state_dict '{}'".format(preTrainedModel))
                state_dict = torch.load(preTrainedModel)
                model_ft.load_state_dict(state_dict)
                print("=> loaded state_dict '{}'".format(preTrainedModel))
            else:
                sys.exit("state_dict not found. Cannot load pretrained model.")

        elif datasetName == 'MSCOCO':
            preTrainedModel = RootPath + 'FT_Models/ResNet50_coco.pth.tar'
            if not preTrainedModel is None and os.path.isfile(preTrainedModel):
                print("=> loading state_dict '{}'".format(preTrainedModel))
                state_dict = torch.load(preTrainedModel)
                model_ft.load_state_dict(state_dict)
                print("=> loaded state_dict '{}'".format(preTrainedModel))
            else:
                sys.exit("state_dict not found. Cannot load pretrained model.")
    
    for p in model_ft.parameters():
        p.requires_grad = False
    
    model_ft.eval()
    
    arch = model_ft.__class__.__name__
    print(arch)
    
    if use_gpu:
        model_ft = model_ft.to(compute_device)

    return model_ft, compute_device 
    
def load_effnet(RootPath, datasetName='PASCAL', model_path=None, device=None):
    use_gpu = torch.cuda.is_available()

    if device is None:
        gpu_info = nvgpu.gpu_info()  
        compute_device = torch.device("cpu")
        for i in range(len(gpu_info)):
            print("MemTotal: ", gpu_info[i]["mem_total"], "MemUsed: ", gpu_info[i]["mem_used"])
            if gpu_info[i]["mem_total"] - gpu_info[i]["mem_used"] > 3000:
                compute_device = torch.device("cuda:"+str(i))
                break
    else:
        compute_device=device	
    if datasetName == 'MSCOCO':
        numClass = 80
    elif datasetName=='PASCAL' or datasetName == 'PASCAL12':
        numClass = 20
        
    if datasetName == 'PASCAL12':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        num_ftrs = model._fc.in_features

        model._fc = nn.Linear(num_ftrs, numClass)

        preTrainedModel = RootPath + 'Models/efficientnetb3_best_checkpoint.pth.tar'

        if not preTrainedModel is None and os.path.isfile(preTrainedModel):
            print("=> loading state_dict '{}'".format(preTrainedModel))
            state_dict = torch.load(preTrainedModel, map_location='cpu')
            model.load_state_dict(state_dict["state_dict"])
            print("=> loaded state_dict '{}'".format(preTrainedModel))
        else:
            sys.exit("state_dict not found. Cannot load pretrained model.")

        for p in model.parameters():
            p.requires_grad = False
        
        model.eval()
        
        if use_gpu:
            model = model.to(compute_device)

        return model, compute_device

    else:
        model = EfficientNet.from_pretrained('efficientnet-b3')

        model.features = model.extract_features

        preTrainedModel = None

        for p in model.parameters():
            p.requires_grad = False
        
        model.eval()
        
        if use_gpu:
            model = model.to(compute_device)

        return model, compute_device
