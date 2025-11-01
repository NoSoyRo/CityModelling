#!/usr/bin/env python3
"""
Quick test script to validate our image loading with the Querétaro dataset.
"""

import sys
from pathlib import Path
import os

# Add src to path correctly
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Change to project directory
os.chdir(current_dir)

from tesis_ac.utils.io import load_image_series, get_image_stats, validate_image_series
from tesis_ac.config import load_config

def main():
    print("🚀 Testing image loading with Querétaro dataset...")
    
    # Load config
    config_path = current_dir / "configs" / "default.yaml"
    config = load_config(config_path)
    
    # Load first few images to test
    data_path = config.data_raw
    print(f"📁 Loading images from: {data_path}")
    
    try:
        # Load just a few images for testing (to avoid memory issues)
        images = load_image_series(data_path, "qrtro_12_198*.jpg")  # Just 1980s
        
        print(f"✅ Loaded {len(images)} images")
        
        # Get statistics
        stats = get_image_stats(images)
        print(f"📊 Years: {stats['years']}")
        print(f"📊 Image shape: {stats['image_shape']}")
        print(f"📊 Data type: {stats['dtype']}")
        print(f"📊 Value range: {stats['value_range']}")
        
        # Validate
        if validate_image_series(images):
            print("✅ Image series validation passed!")
        else:
            print("❌ Image series validation failed!")
            
        # Show first image info
        first_year = min(images.keys())
        first_img = images[first_year]
        print(f"🖼️ First image ({first_year}): shape={first_img.shape}, dtype={first_img.dtype}")
        
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return 1
    
    print("🎉 Image loading test completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())