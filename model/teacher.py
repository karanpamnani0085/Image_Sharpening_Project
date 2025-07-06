import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class TeacherModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(TeacherModel, self).__init__()
        
        # Initialize VGG19 with current best practices
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Use only the features we need to match student model
        layers = list(vgg.features.children())[:20]  # First 20 layers to match size
        self.features = nn.Sequential(*layers)
        
        # Freeze VGG19 parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.conv_out = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        feat = self.features(x)
        return torch.tanh(self.conv_out(feat))