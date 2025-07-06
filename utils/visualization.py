import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_training_progress(losses, metrics, save_path):
    """
    Plot and save training progress
    Args:
        losses (dict): Dictionary containing loss values
        metrics (dict or None): Dictionary containing metric values or None
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    ax1 = plt.subplot(121)
    for loss_name, loss_values in losses.items():
        if loss_values:  # Only plot if we have values
            ax1.plot(loss_values, label=loss_name)
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    
    # Plot metrics if any
    ax2 = plt.subplot(122)
    if metrics is not None:
        for metric_name, metric_values in metrics.items():
            if metric_values:  # Only plot if we have values
                ax2.plot(metric_values, label=metric_name)
        ax2.set_title('Training Metrics')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Metric Value')
        if metrics:  # Only add legend if we have metrics
            ax2.legend()
    else:
        ax2.set_title('No Metrics')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

def compare_images(img1, img2, title1="Image 1", title2="Image 2"):
    """
    Display two images side by side for comparison.
    Accepts torch tensors or numpy arrays.
    """
    import torch

    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
            img = img.numpy()
        # Denormalize if needed (optional, depending on your pipeline)
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img

    img1 = to_numpy(img1)
    img2 = to_numpy(img2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if img1.shape[-1] == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1.squeeze(), cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    if img2.shape[-1] == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2.squeeze(), cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()