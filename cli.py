#!/usr/bin/env python3
"""
Command Line Interface for Satellite Image Classification Pipeline

Usage:
    python cli.py single IMAGE_PATH [OPTIONS]
    python cli.py batch IMAGE_DIR [OPTIONS]
    python cli.py analyze RESULTS_DIR [OPTIONS]

Examples:
    # Process single image
    python cli.py single ../data/raw/imagen_2020.png --output ../results --clusters 2,3,4

    # Process all images in directory
    python cli.py batch ../data/raw --output ../results --pca-components 10

    # Analyze existing results
    python cli.py analyze ../results/satellite_classification_20231101_143022
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tesis_ac.pipeline.main_pipeline import SatelliteImagePipeline, process_image_batch
from tesis_ac.pipeline.visualization import (
    visualize_single_image_results,
    plot_metrics_comparison,
    plot_temporal_comparison,
    create_prediction_map_grid
)


def parse_clusters(clusters_str: str) -> List[int]:
    """Parse cluster string like '2,3,4' into list of integers."""
    try:
        return [int(x.strip()) for x in clusters_str.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid clusters format: {clusters_str}")


def find_images_in_directory(directory: str) -> Dict[str, str]:
    """Find all image files in directory and guess years from filenames."""
    image_dir = Path(directory)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Common image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    images = {}
    for ext in extensions:
        for img_file in image_dir.glob(f"*{ext}"):
            # Try to extract year from filename
            filename = img_file.stem.lower()
            
            # Look for 4-digit years
            import re
            year_match = re.search(r'(19|20)\d{2}', filename)
            if year_match:
                year = year_match.group()
                images[year] = str(img_file)
            else:
                # Use filename as key if no year found
                images[filename] = str(img_file)
    
    return images


def cmd_single(args):
    """Process a single image."""
    print(f"🛰️ Processing single image: {args.image_path}")
    
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return 1
    
    # Create pipeline
    pipeline = SatelliteImagePipeline(
        n_clusters_list=args.clusters,
        n_pca_components=args.pca_components,
        extract_full_features=not args.basic_features,
        svm_kernel=args.svm_kernel,
        random_state=args.random_state
    )
    
    # Process image
    try:
        pipeline.process_image(args.image_path, sample_size=args.sample_size)
        
        # Get summary
        summary = pipeline.get_summary()
        print("\n📊 PROCESSING SUMMARY:")
        print("-" * 40)
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Save results
        output_dir = pipeline.save_results(
            args.output, 
            save_models=args.save_models,
            save_features=args.save_features
        )
        print(f"\n💾 Results saved to: {output_dir}")
        
        # Create visualizations if requested
        if args.visualize:
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
            
            create_prediction_map_grid(
                pipeline.prediction_maps,
                pipeline.image,
                save_path=f"{output_dir}/prediction_maps_grid.png"
            )
        
        print("✅ Single image processing completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return 1


def cmd_batch(args):
    """Process multiple images in batch."""
    print(f"🛰️ Processing images in batch from: {args.image_dir}")
    
    # Find images
    try:
        image_paths = find_images_in_directory(args.image_dir)
    except FileNotFoundError as e:
        print(f"❌ {str(e)}")
        return 1
    
    if not image_paths:
        print(f"❌ No images found in: {args.image_dir}")
        return 1
    
    print(f"📂 Found {len(image_paths)} images:")
    for name, path in image_paths.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   {name}: {os.path.basename(path)} ({size_mb:.1f} MB)")
    
    # Create main pipeline for batch processing
    pipeline = SatelliteImagePipeline(
        n_clusters_list=args.clusters,
        n_pca_components=args.pca_components,
        extract_full_features=not args.basic_features,
        svm_kernel=args.svm_kernel,
        random_state=args.random_state
    )
    
    # Process batch with individual visualizations
    try:
        print(f"\n🚀 Processing {len(image_paths)} images...")
        
        batch_results = pipeline.process_multiple_images(
            image_paths,
            sample_size=args.sample_size,
            visualize=args.visualize,  # This will create individual comparison images
            visualization_style='comprehensive'
        )
        
        # Create output directory for batch results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output) / f"batch_processing_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch results JSON
        with open(output_dir / "batch_results.json", 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"💾 Batch results saved to: {output_dir}")
        
        # Display summary
        print("\n📊 BATCH PROCESSING SUMMARY:")
        print("-" * 50)
        for name, result in batch_results.items():
            if 'error' in result:
                print(f"❌ {name}: {result['error']}")
            else:
                n_features = result.get('n_features', 0)
                pca_var = result.get('pca_variance_explained', 0)
                print(f"✅ {name}: {n_features} features, PCA variance: {pca_var:.3f}")
                
                # Show best accuracies
                classification = result.get('metrics', {}).get('classification', {})
                if classification:
                    best_acc = max(data['accuracy'] for data in classification.values() if 'accuracy' in data)
                    print(f"      Best SVM accuracy: {best_acc:.3f}")
        
        # Create temporal comparison if multiple years and visualization enabled
        successful_years = [name for name in batch_results.keys() if 'error' not in batch_results[name]]
        
        if args.visualize and len(successful_years) > 1:
            print("\n📈 Creating temporal comparison...")
            
            # Create simple temporal plot
            import matplotlib.pyplot as plt
            years_sorted = sorted([y for y in successful_years if y.isdigit()])
            
            if len(years_sorted) > 1:
                vegetation_data = []
                for year in years_sorted:
                    # Extract some metric for temporal comparison
                    metrics = batch_results[year].get('metrics', {})
                    classification = metrics.get('classification', {})
                    if classification:
                        avg_acc = sum(data['accuracy'] for data in classification.values()) / len(classification)
                        vegetation_data.append(avg_acc)
                    else:
                        vegetation_data.append(0)
                
                plt.figure(figsize=(10, 6))
                plt.plot(years_sorted, vegetation_data, 'g-o', linewidth=3, markersize=8)
                plt.title('Temporal Evolution: Average SVM Accuracy')
                plt.xlabel('Year')
                plt.ylabel('Average Accuracy')
                plt.grid(True, alpha=0.3)
                
                temporal_path = output_dir / "temporal_comparison.png"
                plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"   📊 Temporal comparison saved to: {temporal_path}")
        
        print("✅ Batch processing completed!")
        print(f"\n📁 Generated files:")
        if args.visualize:
            print(f"   - comparison_XXXX.png (individual comparison images)")
        print(f"   - batch_results.json (processing summary)")
        if args.visualize and len(successful_years) > 1:
            print(f"   - temporal_comparison.png (temporal analysis)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_analyze(args):
    """Analyze existing results."""
    print(f"🔍 Analyzing results from: {args.results_dir}")
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        return 1
    
    try:
        # Try to load as single image results
        try:
            pipeline = SatelliteImagePipeline.load_results(args.results_dir)
            
            print("\n📊 SINGLE IMAGE RESULTS:")
            print("-" * 40)
            
            summary = pipeline.get_summary()
            for key, value in summary.items():
                print(f"   {key}: {value}")
            
            if pipeline.prediction_maps:
                print(f"\n🗺️  Available prediction maps:")
                for map_name in pipeline.prediction_maps.keys():
                    print(f"   - {map_name}")
            
            # Create visualizations if requested
            if args.visualize and pipeline.image is not None:
                print("\n📈 Creating visualizations...")
                
                output_base = Path(args.results_dir)
                
                visualize_single_image_results(
                    pipeline.image,
                    pipeline.prediction_maps,
                    pipeline.metrics,
                    save_path=str(output_base / "analysis_classification_results.png")
                )
                
                plot_metrics_comparison(
                    pipeline.metrics,
                    save_path=str(output_base / "analysis_metrics_comparison.png")
                )
            
        except:
            # Try to load as batch results
            batch_file = Path(args.results_dir) / "batch_results.json"
            if batch_file.exists():
                with open(batch_file, 'r') as f:
                    batch_results = json.load(f)
                
                print("\n📊 BATCH PROCESSING RESULTS:")
                print("-" * 40)
                
                for name, result in batch_results.items():
                    if 'error' in result:
                        print(f"❌ {name}: {result['error']}")
                    else:
                        n_features = result.get('n_features', 0)
                        pca_var = result.get('pca_variance_explained', 0)
                        print(f"✅ {name}: {n_features} features, PCA variance: {pca_var:.3f}")
                
                # Create temporal comparison if requested
                if args.visualize:
                    successful_years = [name for name in batch_results.keys() if 'error' not in batch_results[name]]
                    if len(successful_years) > 1:
                        print("\n📈 Creating temporal comparison...")
                        plot_temporal_comparison(
                            batch_results,
                            save_path=str(Path(args.results_dir) / "analysis_temporal_comparison.png")
                        )
            else:
                print("❌ Could not identify results format")
                return 1
        
        print("✅ Analysis completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing results: {str(e)}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Satellite Image Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single image command
    single_parser = subparsers.add_parser('single', help='Process a single image')
    single_parser.add_argument('image_path', help='Path to the image file')
    single_parser.add_argument('--output', '-o', default='../data/processed', 
                              help='Output directory (default: ../data/processed)')
    single_parser.add_argument('--clusters', '-c', type=parse_clusters, default=[2, 3, 4],
                              help='Cluster numbers to try (default: 2,3,4)')
    single_parser.add_argument('--pca-components', '-p', type=int, default=8,
                              help='Number of PCA components (default: 8)')
    single_parser.add_argument('--sample-size', '-s', type=int, default=5000,
                              help='Sample size for SVM training (default: 5000)')
    single_parser.add_argument('--basic-features', action='store_true',
                              help='Extract only basic features (faster)')
    single_parser.add_argument('--svm-kernel', choices=['linear', 'rbf', 'poly'], default='linear',
                              help='SVM kernel type (default: linear)')
    single_parser.add_argument('--random-state', type=int, default=42,
                              help='Random state for reproducibility (default: 42)')
    single_parser.add_argument('--save-models', action='store_true',
                              help='Save trained models')
    single_parser.add_argument('--save-features', action='store_true',
                              help='Save feature matrices')
    single_parser.add_argument('--visualize', action='store_true',
                              help='Create visualization plots')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('image_dir', help='Directory containing images')
    batch_parser.add_argument('--output', '-o', default='../data/processed',
                             help='Output directory (default: ../data/processed)')
    batch_parser.add_argument('--clusters', '-c', type=parse_clusters, default=[2, 3, 4],
                             help='Cluster numbers to try (default: 2,3,4)')
    batch_parser.add_argument('--pca-components', '-p', type=int, default=8,
                             help='Number of PCA components (default: 8)')
    batch_parser.add_argument('--sample-size', '-s', type=int, default=5000,
                             help='Sample size for SVM training (default: 5000)')
    batch_parser.add_argument('--basic-features', action='store_true',
                             help='Extract only basic features (faster)')
    batch_parser.add_argument('--svm-kernel', choices=['linear', 'rbf', 'poly'], default='linear',
                             help='SVM kernel type (default: linear)')
    batch_parser.add_argument('--random-state', type=int, default=42,
                             help='Random state for reproducibility (default: 42)')
    batch_parser.add_argument('--visualize', action='store_true',
                             help='Create comprehensive comparison visualizations')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing results')
    analyze_parser.add_argument('results_dir', help='Directory containing results')
    analyze_parser.add_argument('--visualize', action='store_true',
                               help='Create visualization plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to appropriate command
    if args.command == 'single':
        return cmd_single(args)
    elif args.command == 'batch':
        return cmd_batch(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())