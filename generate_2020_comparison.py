#!/usr/bin/env python3
"""
Generate comparison for 2020 image only.
"""

import sys
from pathlib import Path
import gc

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from tesis_ac.pipeline.main_pipeline import SatelliteImagePipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Generate comparison for 2020."""
    print("🛰️ Processing 2020: data/raw/imagen_2020.png")
    
    try:
        # Create pipeline with optimized settings
        pipeline = SatelliteImagePipeline(
            n_clusters_list=[2, 3, 4],
            n_pca_components=8,
            extract_full_features=True,
            svm_kernel='linear',
            random_state=42
        )
        
        # Process image with smaller sample for SVM to speed up
        pipeline.process_image('data/raw/imagen_2020.png', sample_size=3000)
        
        # Create comprehensive visualization
        pipeline.visualize_results(
            save_path='comparison_2020.png',
            style='comprehensive',
            year='2020'
        )
        
        print("✅ 2020 completed successfully!")
        print("📊 Generated: comparison_2020.png")
        
        # Print summary
        metrics = pipeline.metrics
        classification = metrics.get('classification', {})
        clustering = metrics.get('clustering', {})
        
        print(f"📈 Summary for 2020:")
        print(f"   Features: {len(pipeline.feature_names)}")
        print(f"   PCA variance: {pipeline.preprocessor.get_variance_explained():.3f}")
        
        for n_clusters, data in classification.items():
            acc = data.get('accuracy', 0)
            inertia = clustering.get(n_clusters, {}).get('inertia', 0)
            print(f"   {n_clusters} clusters: SVM accuracy={acc:.3f}, inertia={inertia:.1f}")
        
        # Clean up memory
        del pipeline
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing 2020: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    
    # Show final results
    print(f"\n🎉 FINAL RESULTS:")
    for comparison_file in Path('.').glob('comparison_*.png'):
        size_mb = comparison_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ {comparison_file.name} ({size_mb:.1f} MB)")