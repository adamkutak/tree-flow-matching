import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_fid import fid_score
import tempfile
import os

def visualize_samples(samples, title="Generated Samples", nrow=10):
    """Visualize a grid of samples."""
    grid = make_grid(
        samples.clip(-1, 1),
        value_range=(-1, 1),
        padding=0,
        nrow=nrow
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(ToPILImage()(grid))
    plt.axis('off')
    plt.title(title)
    plt.show()

def calculate_diversity_metrics(samples):
    """Calculate metrics for sample diversity."""
    # Flatten samples
    flat_samples = samples.view(samples.size(0), -1)
    
    # Calculate pairwise distances
    distances = torch.pdist(flat_samples)
    avg_distance = distances.mean()
    std_distance = distances.std()
    
    # Calculate number of unique patterns (with some tolerance)
    tolerance = 1e-5
    unique_patterns = torch.unique(torch.round(flat_samples / tolerance) * tolerance, dim=0).shape[0]
    
    return {
        'avg_distance': avg_distance.item(),
        'std_distance': std_distance.item(),
        'unique_patterns': unique_patterns,
        'coverage': unique_patterns / samples.size(0)
    }

def calculate_fid(samples, test_loader, num_samples, device):
    """Calculate FID score between real and generated samples."""
    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
        # Save real images
        for i, (imgs, _) in enumerate(test_loader):
            if i * test_loader.batch_size >= num_samples:
                break
            for j, img in enumerate(imgs):
                save_image(img, os.path.join(real_dir, f'{i*test_loader.batch_size+j}.png'))
        
        # Save generated images
        for i, img in enumerate(samples):
            save_image(img, os.path.join(fake_dir, f'{i}.png'))
        
        # Calculate FID
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
    return fid

def visualize_class_samples(samples, class_labels):
    """Visualize samples grouped by class."""
    class_samples = {i: [] for i in range(10)}
    for sample, label in zip(samples, class_labels):
        class_samples[label.item()].append(sample)
    
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        if class_samples[i]:
            samples_grid = make_grid(
                torch.stack(class_samples[i][:16]), 
                nrow=4,
                normalize=True
            )
            plt.imshow(ToPILImage()(samples_grid))
        plt.title(f"Digit {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()