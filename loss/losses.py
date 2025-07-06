import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from math import exp

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Use fewer layers to maintain larger feature maps
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:20]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # Ensure input sizes match
        if x.size() != y.size():
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        
        return F.mse_loss(x_features, y_features)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self.create_window()
        
    def gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self):
        _1D_window = self.gaussian(self.window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def forward(self, img1, img2):
        # Ensure input sizes match
        if img1.size() != img2.size():
            img2 = F.interpolate(img2, size=img1.size()[2:], mode='bilinear', align_corners=False)
        
        # Move window to same device as input
        self.window = self.window.to(img1.device)
        
        # Compute SSIM
        mu1 = F.conv2d(img1, self.window.expand(3, 1, -1, -1).contiguous(), padding=self.window_size//2, groups=3)
        mu2 = F.conv2d(img2, self.window.expand(3, 1, -1, -1).contiguous(), padding=self.window_size//2, groups=3)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window.expand(3, 1, -1, -1).contiguous(), padding=self.window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window.expand(3, 1, -1, -1).contiguous(), padding=self.window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window.expand(3, 1, -1, -1).contiguous(), padding=self.window_size//2, groups=3) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_outputs, teacher_outputs):
        """
        Compute the knowledge distillation loss between student and teacher outputs
        Args:
            student_outputs: output features from the student model
            teacher_outputs: output features from the teacher model
        """
        # Ensure inputs have the same size
        if student_outputs.size() != teacher_outputs.size():
            teacher_outputs = F.interpolate(
                teacher_outputs,
                size=student_outputs.size()[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Reshape tensors to 2D for softmax
        b, c, h, w = student_outputs.size()
        student_outputs = student_outputs.view(b, c, -1)
        teacher_outputs = teacher_outputs.view(b, c, -1)
        
        # Scale outputs by temperature
        student_outputs = student_outputs / self.temperature
        teacher_outputs = teacher_outputs.detach() / self.temperature
        
        # Compute KL divergence loss
        loss = F.kl_div(
            F.log_softmax(student_outputs, dim=1),
            F.softmax(teacher_outputs, dim=1),
            reduction='batchmean'
        )
        
        # Scale loss back with temperature
        return loss * (self.temperature ** 2)