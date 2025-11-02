#!/usr/bin/env python3
"""
Generate individual comprehensive comparison images for each raw image.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from tesis_ac.pipeline.main_pipeline import SatelliteImagePipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_single_image(image_path, year):
    """Process single image and create comprehensive comparison."""
    print(f"\n🛰️ Processing {year}: {image_path}")
    
    try:
        # Create pipeline
        pipeline = SatelliteImagePipeline(
            n_clusters_list=[2, 3, 4],
            n_pca_components=8,
            extract_full_features=True,
            svm_kernel='linear',
            random_state=42
        )
        
        # Process image
        pipeline.process_image(image_path, sample_size=5000)
        
        # Create comprehensive visualization
        viz_path = f"comparison_{year}.png"
        pipeline.visualize_results(
            save_path=viz_path,
            style='comprehensive',
            year=year
        )
        
        print(f"✅ {year} completed successfully!")
        print(f"📊 Generated: {viz_path}")
        
        # Print summary
        metrics = pipeline.metrics
        classification = metrics.get('classification', {})
        clustering = metrics.get('clustering', {})
        
        print(f"📈 Summary for {year}:")
        print(f"   Features: {len(pipeline.feature_names)}")
        print(f"   PCA variance: {pipeline.preprocessor.get_variance_explained():.3f}")
        
        for n_clusters, data in classification.items():
            acc = data.get('accuracy', 0)
            inertia = clustering.get(n_clusters, {}).get('inertia', 0)
            print(f"   {n_clusters} clusters: SVM accuracy={acc:.3f}, inertia={inertia:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {year}: {str(e)}")
        return False

def main():
    """Main function."""
    print("🚀 Generating comprehensive comparison images for all years")
    print("=" * 60)
    
    # Define images
    images = {
        '2003': 'data/raw/imagen_2003.png',
        '2020': 'data/raw/imagen_2020.png'
    }
    
    # Process each image
    successful = 0
    for year, path in images.items():
        if Path(path).exists():
            if process_single_image(path, year):
                successful += 1
        else:
            print(f"❌ {year}: File not found: {path}")
    
    print(f"\n🎉 COMPLETED: {successful}/{len(images)} images processed")
    print(f"📁 Generated comparison images:")
    for comparison_file in Path('.').glob('comparison_*.png'):
        size_mb = comparison_file.stat().st_size / (1024 * 1024)
        print(f"   {comparison_file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()