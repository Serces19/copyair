
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.manifold import MDS
import glob

# Add src to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.losses import PerceptualLoss

def load_image(path, size=(256, 256)):
    """Load image, resize and normalize to [-1, 1]"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1] -> [-1, 1]
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0) # Add batch dimension

def add_noise(img_tensor, noise_level=0.1):
    """Add Gaussian noise to the image"""
    noise = torch.randn_like(img_tensor) * noise_level
    noisy_img = img_tensor + noise
    return torch.clamp(noisy_img, -1, 1)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Perceptual Loss
    # net='alex' is standard for LPIPS, 'vgg' is also common
    lpips = PerceptualLoss(net='alex', device=device)
    
    # Load images
    data_dir = os.path.join('data', '03_processed', 'gt')
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    
    if not image_paths:
        print(f"No images found in {data_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    
    # Limit to first 10 images for visualization clarity if there are many
    selected_paths = image_paths[:10]
    images = [load_image(p).to(device) for p in selected_paths]
    image_names = [os.path.basename(p) for p in selected_paths]

    # --- Experiment 1: Pairwise Distance Matrix ---
    print("\n--- Computing Pairwise Distance Matrix ---")
    n_imgs = len(images)
    dist_matrix = np.zeros((n_imgs, n_imgs))

    with torch.no_grad():
        for i in range(n_imgs):
            for j in range(n_imgs):
                if i == j:
                    dist_matrix[i, j] = 0
                elif i > j: # Symmetric
                    dist_matrix[i, j] = dist_matrix[j, i]
                else:
                    d = lpips(images[i], images[j])
                    dist_matrix[i, j] = d.item()

    # Plot Distance Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(dist_matrix, cmap='viridis')
    plt.colorbar(label='LPIPS Distance')
    plt.xticks(range(n_imgs), image_names, rotation=45, ha='right')
    plt.yticks(range(n_imgs), image_names)
    plt.title('Pairwise LPIPS Distance Matrix')
    plt.tight_layout()
    plt.savefig('notebooks/lpips_distance_matrix.png')
    print("Saved notebooks/lpips_distance_matrix.png")

    # --- Experiment 2: Noise Sensitivity ---
    print("\n--- Testing Noise Sensitivity ---")
    base_img = images[0]
    noise_levels = np.linspace(0, 1.0, 20)
    distances = []

    with torch.no_grad():
        for nl in noise_levels:
            noisy = add_noise(base_img, noise_level=nl)
            d = lpips(base_img, noisy)
            distances.append(d.item())

    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, distances, marker='o')
    plt.xlabel('Gaussian Noise Level (std dev)')
    plt.ylabel('LPIPS Distance')
    plt.title(f'LPIPS Sensitivity to Noise ({image_names[0]})')
    plt.grid(True)
    plt.savefig('notebooks/lpips_noise_sensitivity.png')
    print("Saved notebooks/lpips_noise_sensitivity.png")

    # --- Experiment 3: 2D Visualization (MDS) ---
    print("\n--- Generating 2D Visualization (MDS) ---")
    # We use the distance matrix computed earlier
    # MDS tries to place points in 2D such that their Euclidean distances preserve the input dissimilarities (LPIPS)
    
    # Let's add some noisy variants to the map to see if they drift away
    # We'll take the first 3 images and generate 3 noisy versions for each
    
    viz_images = []
    viz_labels = []
    viz_colors = []
    
    # Colors for different source images
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i in range(min(3, len(images))):
        img = images[i]
        name = image_names[i]
        c = colors[i % len(colors)]
        
        # Original
        viz_images.append(img)
        viz_labels.append(f"{name} (Orig)")
        viz_colors.append(c)
        
        # Noisy versions
        for nl in [0.1, 0.3, 0.5]:
            noisy = add_noise(img, noise_level=nl)
            viz_images.append(noisy)
            viz_labels.append(f"{name} (N={nl})")
            viz_colors.append(c) # Same color family
            
    # Compute new distance matrix for these viz_images
    n_viz = len(viz_images)
    viz_dist_matrix = np.zeros((n_viz, n_viz))
    
    with torch.no_grad():
        for i in range(n_viz):
            for j in range(i, n_viz):
                d = lpips(viz_images[i], viz_images[j])
                viz_dist_matrix[i, j] = d.item()
                viz_dist_matrix[j, i] = d.item()
                
    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(viz_dist_matrix)
    
    plt.figure(figsize=(10, 8))
    
    # Plot points
    for i in range(n_viz):
        # Use different markers for original vs noisy?
        marker = 'o' if 'Orig' in viz_labels[i] else 'x'
        size = 100 if 'Orig' in viz_labels[i] else 50
        alpha = 1.0 if 'Orig' in viz_labels[i] else 0.7
        
        plt.scatter(coords[i, 0], coords[i, 1], c=viz_colors[i], marker=marker, s=size, alpha=alpha, label=viz_labels[i] if 'Orig' in viz_labels[i] else "")
        
        # Annotate
        plt.annotate(viz_labels[i], (coords[i, 0], coords[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title('2D Map of Image Similarity (LPIPS + MDS)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='gray', markersize=10),
                       Line2D([0], [0], marker='x', color='w', label='Noisy', markeredgecolor='gray', markersize=10)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('notebooks/lpips_2d_map.png')
    print("Saved notebooks/lpips_2d_map.png")
    
    print("\nDone! Check the 'notebooks' directory for the generated plots.")

if __name__ == "__main__":
    main()
