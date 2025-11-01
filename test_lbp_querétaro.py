#!/usr/bin/env python3
"""
Test script para probar extracción de LBP con imágenes de Querétaro.
"""

import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

# Add src to path correctly
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))
os.chdir(current_dir)

from tesis_ac.utils.io import load_image_series, get_image_stats
from tesis_ac.features.lbp import compute_lbp, compute_lbp_statistics, normalize_lbp
from tesis_ac.config import load_config

def main():
    print("🚀 Testing LBP extraction with Querétaro dataset...")
    
    # Load config
    config_path = current_dir / "configs" / "default.yaml"
    config = load_config(config_path)
    
    # Load first image for testing
    data_path = config.data_raw
    print(f"📁 Loading images from: {data_path}")
    
    try:
        # Load just one image for testing
        images = load_image_series(data_path, "qrtro_12_1984.jpg")
        
        if not images:
            print("❌ No images loaded")
            return 1
            
        # Get first image
        year, image = next(iter(images.items()))
        print(f"🖼️ Testing with image from year {year}")
        print(f"📊 Image shape: {image.shape}")
        
        # Extract LBP features
        print("🔍 Computing LBP features...")
        lbp_params = config.features.lbp
        lbp_features = compute_lbp(
            image, 
            P=lbp_params['P'], 
            R=lbp_params['R'], 
            method=lbp_params['method']
        )
        
        print(f"✅ LBP shape: {lbp_features.shape}")
        
        # Get statistics
        lbp_stats = compute_lbp_statistics(lbp_features)
        print(f"📊 LBP Statistics:")
        for key, value in lbp_stats.items():
            print(f"   {key}: {value}")
        
        # Normalize features
        lbp_normalized = normalize_lbp(lbp_features)
        print(f"✅ Normalized LBP range: [{lbp_normalized.min():.3f}, {lbp_normalized.max():.3f}]")
        
        # Create visualization
        print("📊 Creating visualization...")
        create_lbp_visualization(image, lbp_features, lbp_normalized, year)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("🎉 LBP extraction test completed successfully!")
    return 0

def create_lbp_visualization(original, lbp_raw, lbp_norm, year):
    """Create a visualization of original image and LBP features."""
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'LBP Features - Querétaro {year}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original RGB')
    axes[0, 0].axis('off')
    
    # LBP for each channel (raw)
    for i in range(3):
        ax = axes[0, i] if i == 0 else axes[1, i-1] if i < 3 else axes[1, 2]
        if i == 0:
            continue  # Skip first position (used for original)
            
        channel_idx = i - 1
        im = ax.imshow(lbp_raw[:, :, channel_idx], cmap='gray')
        ax.set_title(f'LBP Channel {channel_idx} (Raw)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # LBP channels normalized
    for i in range(3):
        ax = axes[1, i]
        im = ax.imshow(lbp_norm[:, :, i], cmap='viridis')
        ax.set_title(f'LBP Channel {i} (Normalized)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = current_dir / "figs" / f"lbp_test_{year}.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved to: {output_path}")
    
    # Also show
    plt.show()

if __name__ == "__main__":
    exit(main())