import os
import numpy as np
from skimage import io, filters
from skimage.util import random_noise

def create_sample_dataset(num_samples=10):
    """Generate sample image pairs for training"""
    # Create directories if they don't exist
    os.makedirs('data/train/input', exist_ok=True)
    os.makedirs('data/train/target', exist_ok=True)
    
    # Generate sample images
    for i in range(num_samples):
        # Create a random sharp image
        sharp = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Create blurred version
        blurred = filters.gaussian(sharp, sigma=2, channel_axis=2)
        blurred = (blurred * 255).astype(np.uint8)
        
        # Save images
        io.imsave(f'data/train/target/sample_{i}.png', sharp)
        io.imsave(f'data/train/input/sample_{i}.png', blurred)

if __name__ == '__main__':
    create_sample_dataset()
    print("Sample dataset created successfully!")