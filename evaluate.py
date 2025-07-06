import torch
from torch.utils.data import DataLoader
from utils.dataset import ImageSharpeningDataset
from utils.tools import load_checkpoint, get_device
from model.student import StudentModel
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def tensor_to_image(tensor):
    # Assumes tensor is 1x3xHxW or 3xHxW, normalized
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    tensor = torch.clamp(tensor, 0, 1)
    img = (tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return img

def evaluate(model, val_loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            # Convert tensors to images for metric calculation
            output_img = tensor_to_image(outputs)
            target_img = tensor_to_image(targets)
            # Compute PSNR and SSIM
            psnr = compare_psnr(target_img, output_img, data_range=255)
            # Use channel_axis=-1 for color images (scikit-image >= 0.19)
            min_side = min(target_img.shape[0], target_img.shape[1])
            win_size = min(7, min_side) if min_side % 2 == 1 else min(7, min_side - 1)
            ssim = compare_ssim(target_img, output_img, data_range=255, channel_axis=-1, win_size=win_size)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    return total_psnr / count, total_ssim / count

def main():
    # Load config
    with open('config.json') as f:
        config = json.load(f)
    
    device = get_device()
    
    # Load model
    model = StudentModel(config['model']['in_channels'],
                        config['model']['out_channels'],
                        config['model']['student_features']).to(device)
    
    # Load checkpoint
    load_checkpoint('checkpoints/student_epoch_50.pth', model)
    
    # Data loading
    val_dataset = ImageSharpeningDataset(config['data']['val_data_dir'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Evaluate
    psnr, ssim = evaluate(model, val_loader, device)
    print(f"Evaluation Results:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

if __name__ == '__main__':
    main()