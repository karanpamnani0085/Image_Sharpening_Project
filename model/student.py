import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=32):  # Changed default features to 32
        super(StudentModel, self).__init__()
        
        # Lightweight encoder for real-time processing
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Efficient sharpening module
        self.sharpening = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        features = self.encoder(x)
        sharpened = self.sharpening(features)
        return x + sharpened