import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageSharpeningDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.input_dir = os.path.join(data_dir, "input")
        self.target_dir = os.path.join(data_dir, "target")
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),  # Reduce image size for less GPU memory
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_files = [
            f for f in os.listdir(self.input_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.input_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        input_path = os.path.join(self.input_dir, img_name)
        input_image = Image.open(input_path).convert('RGB')
        
        target_path = os.path.join(self.target_dir, img_name)
        target_image = Image.open(target_path).convert('RGB')
        
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)
        
        return input_tensor, target_tensor