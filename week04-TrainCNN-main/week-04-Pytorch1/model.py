import torch
import torch.nn as nn
import torchvision

class vanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.cv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.cv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout()
        self.head = nn.Linear(in_features=9216, out_features=20)
    
    def forward(self, x):
        #################### fill here #####################
        #   TODO: forward()를 정의해주세요.
        #   cv1 -> relu -> pool1 -> cv2 -> relu -> pool2
        #   -> cv3 -> relu -> cv4 -> relu -> cv5 -> relu -> 
        #   pool3 -> dropout -> head 순서로 거쳐야 합니다.
        ####################################################
        x1 = self.cv1(x)   
        x2 = self.relu(x1)
        x3 = self.pool1(x2)
        x4 = self.cv2(x3)
        x5 = self.relu(x4)
        x6 = self.pool2(x5)
        x7 = self.cv3(x6)
        x8 = self.relu(x7)
        x9 = self.cv4(x8)
        x10 = self.relu(x9)
        x11 = self.cv5(x10)
        x12 = self.relu(x11)
        x13 = self.pool3(x12)
        x14 = self.dropout(x13)
        x15 = x14.view(x14.size(0), -1)
        x16 = self.head(x15)
        return x16

class vanillaCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.cv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.cv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout()
        
        ################### fill here #####################
        #   TODO: MLP head (self.head)를 정의해주세요
        ###################################################
        self.head = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=20)
        )
    def forward(self, x):
        ################### fill here #####################
        #   TODO: forward()를 정의해주세요
        #   vanillaCNN 과 동일하게 사용해도 무방합니다.
        ###################################################
        x1 = self.cv1(x)   
        x2 = self.relu(x1)
        x3 = self.pool1(x2)
        x4 = self.cv2(x3)
        x5 = self.relu(x4)
        x6 = self.pool2(x5)
        x7 = self.cv3(x6)
        x8 = self.relu(x7)
        x9 = self.cv4(x8)
        x10 = self.relu(x9)
        x11 = self.cv5(x10)
        x12 = self.relu(x11)
        x13 = self.pool3(x12)
        x14 = self.dropout(x13)
        x15 = x14.view(x14.size(0), -1)
        x16 = self.head(x15)
        return x16
    
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("loading Imagenet pretrained VGG19")
        self.vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1', progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=20)
        )
        # replace classifier of pretrained VGG-19 with self defined classifier
        setattr(self.vgg, 'classifier', self.classifier)
    
    def forward(self, x):
        return self.vgg(x)