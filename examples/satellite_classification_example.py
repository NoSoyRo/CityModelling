"""
Example script showing how to use the satellite image classification pipeline.

This script demonstrates:
1. Processing a single image
2. Processing multiple images in batch
3. Saving and loading results
4. Creating visualizations
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from tesis_ac.pipeline.main_pipeline import SatelliteImagePipeline, process_image_batch
from tesis_ac.pipeline.visualization import (
    visualize_single_image_results, 
    plot_metrics_comparison,
    plot_temporal_comparison
)


def example_single_image():
    """Example: Process a single image."""
    print("=" * 60)
    print("EXAMPLE 1: PROCESSING SINGLE IMAGE")
    print("=" * 60)
    
    # Image path
    image_path = "../data/raw/imagen_2020.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Create pipeline
    pipeline = SatelliteImagePipeline(
        n_clusters_list=[2, 3, 4],
        n_pca_components=8,
        extract_full_features=True,
        svm_kernel='linear'
    )
    
    # Process image
    print(f"🚀 Processing image: {image_path}")
    pipeline.process_image(image_path, sample_size=3000)
    
    # Get summary
    summary = pipeline.get_summary()
    print("\n📊 PROCESSING SUMMARY:")
    print("-" * 40)
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Save results
    output_dir = pipeline.save_results("../data/processed", save_models=True)
    print(f"\n💾 Results saved to: {output_dir}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    visualize_single_image_results(
        pipeline.image,
        pipeline.prediction_maps,
        pipeline.metrics,
        save_path=f"{output_dir}/classification_results.png"
    )
    
    plot_metrics_comparison(
        pipeline.metrics,
        save_path=f"{output_dir}/metrics_comparison.png"
    )
    
    print("✅ Single image processing completed!")
    return pipeline


def example_batch_processing():
    """Example: Process multiple images in batch."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: BATCH PROCESSING MULTIPLE IMAGES")
    print("=" * 60)
    
    # Define image paths
    image_paths = {
        '1984': '../data/raw/imagen_1984.png',
        '2003': '../data/raw/imagen_2003.png', 
        '2020': '../data/raw/imagen_2020.png'
    }
    
    # Check which images exist
    available_images = {}
    for year, path in image_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            available_images[year] = path
            print(f"✅ {year}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {year}: {path} (NOT FOUND)")
    
    if not available_images:
        print("❌ No images found for batch processing")
        return
    
    # Process batch
    print(f"\n🚀 Processing {len(available_images)} images...")
    
    output_dir = process_image_batch(
        available_images,
        "../data/processed",
        n_clusters_list=[2, 3, 4],
        n_pca_components=8,
        sample_size=3000
    )
    
    print(f"💾 Batch results saved to: {output_dir}")
    
    # Load and display results
    import json
    with open(f"{output_dir}/batch_results.json", 'r') as f:
        batch_results = json.load(f)
    
    print("\n📊 BATCH PROCESSING SUMMARY:")
    print("-" * 50)
    for year, result in batch_results.items():
        if 'error' in result:
            print(f"❌ {year}: {result['error']}")
        else:
            n_features = result.get('n_features', 0)
            pca_var = result.get('pca_variance_explained', 0)
            print(f"✅ {year}: {n_features} features, PCA variance: {pca_var:.3f}")
            
            # Show best accuracies
            classification = result.get('metrics', {}).get('classification', {})
            valid_class = {k: v for k, v in classification.items() if 'accuracy' in v}
            if valid_class:
                best_acc = max(data['accuracy'] for data in valid_class.values())
                print(f"      Best SVM accuracy: {best_acc:.3f}")
    
    # Create temporal comparison if we have multiple years
    if len([year for year in batch_results.keys() if 'error' not in batch_results[year]]) > 1:
        print("\n📈 Creating temporal comparison...")
        plot_temporal_comparison(
            batch_results,
            save_path=f"{output_dir}/temporal_comparison.png"
        )
    
    print("✅ Batch processing completed!")
    return output_dir


def example_load_and_analyze():
    """Example: Load previously saved results and analyze."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: LOAD AND ANALYZE SAVED RESULTS")
    print("=" * 60)
    
    # Look for recent results
    processed_dir = Path("../data/processed")
    if not processed_dir.exists():
        print("❌ No processed data directory found")
        return
    
    # Find most recent single image results
    result_dirs = list(processed_dir.glob("satellite_classification_*"))
    if not result_dirs:
        print("❌ No saved results found")
        return
    
    # Load most recent
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading results from: {latest_dir}")
    
    try:
        pipeline = SatelliteImagePipeline.load_results(str(latest_dir))
        
        # Display summary
        summary = pipeline.get_summary()
        print("\n📊 LOADED RESULTS SUMMARY:")
        print("-" * 40)
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Show available prediction maps
        if pipeline.prediction_maps:
            print(f"\n🗺️  Available prediction maps:")
            for map_name in pipeline.prediction_maps.keys():
                print(f"   - {map_name}")
        
        print("✅ Results loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading results: {str(e)}")


def main():
    """Run all examples."""
    print("🛰️ SATELLITE IMAGE CLASSIFICATION PIPELINE EXAMPLES")
    print("=" * 80)
    
    # Example 1: Single image
    try:
        pipeline = example_single_image()
    except Exception as e:
        print(f"❌ Single image example failed: {str(e)}")
        pipeline = None
    
    # Example 2: Batch processing
    try:
        batch_output = example_batch_processing()
    except Exception as e:
        print(f"❌ Batch processing example failed: {str(e)}")
        batch_output = None
    
    # Example 3: Load and analyze
    try:
        example_load_and_analyze()
    except Exception as e:
        print(f"❌ Load and analyze example failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("🎉 EXAMPLES COMPLETED!")
    print("=" * 80)
    
    # Final summary
    print("\n📋 WHAT WAS CREATED:")
    
    if pipeline:
        print("✅ Single image pipeline with trained models")
        print("✅ Visualization plots saved")
    
    if batch_output:
        print("✅ Batch processing results for multiple years")
        print("✅ Temporal comparison analysis")
    
    print("\n📖 NEXT STEPS:")
    print("1. Check the output directories for saved results")
    print("2. Examine the visualization plots")
    print("3. Use the saved models for new predictions")
    print("4. Customize pipeline parameters for your specific needs")


if __name__ == "__main__":
    main()