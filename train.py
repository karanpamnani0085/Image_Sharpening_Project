import torch
import random
import numpy as np
import json
import os
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.dataset import ImageSharpeningDataset
from model.student import StudentModel
from model.teacher import TeacherModel
from loss.losses import PerceptualLoss, SSIMLoss, KnowledgeDistillationLoss
from utils.tools import get_device, save_checkpoint
from utils.visualization import plot_training_progress
import sys

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    # Initialize models
    teacher = TeacherModel(
        config['model']['in_channels'],
        config['model']['out_channels'],
        config['model']['teacher_features']
    ).to(device)
    
    student = StudentModel(
        config['model']['in_channels'],
        config['model']['out_channels'],
        config['model']['student_features']
    ).to(device)
    
    # Initialize dataset and dataloader
    train_dataset = ImageSharpeningDataset(config['data']['train_data_dir'])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,  # Reduce workers to save memory
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize losses
    perceptual_loss = PerceptualLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    kd_loss = KnowledgeDistillationLoss().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(student.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Ensure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    losses = {'total': [], 'perceptual': [], 'ssim': [], 'kd': []}
    scaler = GradScaler()
    try:
        for epoch in range(config['training']['epochs']):
            student.train()
            epoch_losses = {'total': 0, 'perceptual': 0, 'ssim': 0, 'kd': 0}
            batch_count = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                with autocast():
                    student_outputs = student(images)
                    with torch.no_grad():
                        teacher_outputs = teacher(images)
                    p_loss = perceptual_loss(student_outputs, targets)
                    s_loss = ssim_loss(student_outputs, targets)
                    kd_loss_val = kd_loss(student_outputs, teacher_outputs)
                    total_loss = (config['training']['lambda_p'] * p_loss + 
                                  config['training']['lambda_s'] * s_loss +
                                  kd_loss_val)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(total_loss)
                
                # Accumulate losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['perceptual'] += p_loss.item()
                epoch_losses['ssim'] += s_loss.item()
                epoch_losses['kd'] += kd_loss_val.item()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}] '
                          f'Batch [{batch_idx+1}/{len(train_loader)}] '
                          f'Loss: {total_loss.item():.4f}')
            
            # Calculate average losses for the epoch
            for key in losses:
                avg_loss = epoch_losses[key] / batch_count
                losses[key].append(avg_loss)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'checkpoints/student_epoch_{epoch+1}.pth')
                
                # Plot and save progress
                plot_training_progress(
                    losses, 
                    None,  # No validation losses
                    f'checkpoints/training_plot_epoch_{epoch+1}.png'
                )
            # Free up unused GPU memory
            torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print('\nInterrupted! Saving checkpoint before exit...')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'checkpoints/student_interrupted.pth')
        sys.exit(0)

if __name__ == '__main__':
    main()