#!/usr/bin/env python3
"""
CLI interface for the satellite image processing pipeline.
Provides batch processing capabilities for multiple images.
"""

import argparse
import sys
import os
from pathlib import Path
import glob
from datetime import datetime
import json
import logging

# Configure matplotlib to use non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tesis_ac.pipeline.main_pipeline import SatelliteImagePipeline
from tesis_ac.pipeline import visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processor for multiple satellite images."""
    
    def __init__(self, input_dir: str, output_dir: str, config: dict = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline with binary classification (only 2 clusters)
        self.pipeline = SatelliteImagePipeline(
            n_clusters_list=[2],  # Only binary classification
            n_pca_components=8,
            extract_full_features=True,
            svm_kernel='linear',
            random_state=42
        )
        
        # Store visualization module for later use
        self.visualization = visualization
        
        # Track last output directory
        self.last_output_dir = None
    
    def discover_images(self) -> dict:
        """Discover PNG images in input directory."""
        image_paths = {}
        
        # Look for PNG files
        png_files = list(self.input_dir.glob("*.png"))
        
        for png_file in sorted(png_files):
            # Extract year from filename (assuming format: imagen_YYYY.png)
            filename = png_file.stem
            if 'imagen_' in filename:
                try:
                    year = filename.split('_')[-1]
                    if year.isdigit() and len(year) == 4:
                        image_paths[year] = str(png_file)
                        logger.info(f"Found image for year {year}: {png_file.name}")
                except:
                    logger.warning(f"Could not extract year from filename: {filename}")
            else:
                # Use filename as key if no year pattern found
                image_paths[filename] = str(png_file)
                logger.info(f"Found image: {png_file.name}")
        
        logger.info(f"Discovered {len(image_paths)} images")
        return image_paths
    
    def process_all_images(self, sample_size: int = 10000) -> dict:
        """Process all discovered images with incremental saving."""
        image_paths = self.discover_images()
        
        if not image_paths:
            logger.error("No images found in input directory")
            return {}
        
        logger.info(f"Starting batch processing of {len(image_paths)} images...")
        
        # Create timestamped output directory immediately
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = self.output_dir / f"batch_processing_{timestamp}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results incrementally to: {batch_output_dir}")
        
        # Process images one by one and save immediately
        results = {}
        
        for idx, (year, path) in enumerate(sorted(image_paths.items()), 1):
            logger.info(f"Processing {idx}/{len(image_paths)}: {year}")
            
            try:
                # Create new pipeline instance for each image
                pipeline = SatelliteImagePipeline(
                    n_clusters_list=self.pipeline.n_clusters_list,
                    n_pca_components=self.pipeline.n_pca_components,
                    extract_full_features=self.pipeline.extract_full_features,
                    svm_kernel=self.pipeline.svm_kernel,
                    random_state=self.pipeline.random_state
                )
                
                # Process single image
                result = pipeline.process_image(path, sample_size=sample_size)
                
                # Generate simple visualization for this image
                self._create_simple_comparison(result, year, path, batch_output_dir)
                
                # Save individual result immediately
                individual_output_dir = batch_output_dir / f"year_{year}"
                result.save_results(str(individual_output_dir), save_models=False)
                
                # Store for batch summary
                results[year] = result
                
                logger.info(f"✅ Completed and saved: {year} ({idx}/{len(image_paths)})")
                
            except Exception as e:
                logger.error(f"❌ Failed processing {year}: {e}")
                continue
        
        # Save batch summary at the end
        if results:
            self._save_batch_summary(results, batch_output_dir)
        
        # Store output directory for later access
        self.last_output_dir = batch_output_dir
        
        return results
    
    def generate_batch_summary(self, results: dict) -> dict:
        """Generate summary statistics for batch processing."""
        summary = {
            'total_images': len(results),
            'processing_timestamp': datetime.now().isoformat(),
            'years_processed': list(results.keys()),
            'binary_classification_stats': {}
        }
        
        for year, result in results.items():
            if hasattr(result, 'results') and 'predictions' in result.results:
                predictions = result.results['predictions']
                if '2_clusters' in predictions:
                    binary_pred = predictions['2_clusters']['svm']
                    n_urban = int((binary_pred == 1).sum())
                    n_rural = int((binary_pred == 0).sum())
                    total = n_urban + n_rural
                    
                    summary['binary_classification_stats'][year] = {
                        'urban_pixels': n_urban,
                        'rural_pixels': n_rural,
                        'total_pixels': total,
                        'urban_percentage': (n_urban / total * 100) if total > 0 else 0,
                        'rural_percentage': (n_rural / total * 100) if total > 0 else 0
                    }
        
        return summary
    
    def create_temporal_analysis(self, results: dict) -> dict:
        """Create temporal analysis of urban growth."""
        temporal_data = {}
        
        # Extract urban percentages by year
        years = []
        urban_percentages = []
        
        for year in sorted(results.keys()):
            if year.isdigit():
                result = results[year]
                if hasattr(result, 'results') and 'predictions' in result.results:
                    predictions = result.results['predictions']
                    if '2_clusters' in predictions:
                        binary_pred = predictions['2_clusters']['svm']
                        urban_pct = (binary_pred == 1).sum() / len(binary_pred) * 100
                        
                        years.append(int(year))
                        urban_percentages.append(float(urban_pct))
        
        if len(years) > 1:
            # Calculate growth rates
            growth_rates = []
            for i in range(1, len(urban_percentages)):
                rate = urban_percentages[i] - urban_percentages[i-1]
                growth_rates.append(rate)
            
            temporal_data = {
                'years': years,
                'urban_percentages': urban_percentages,
                'growth_rates': growth_rates,
                'total_growth_1984_2020': urban_percentages[-1] - urban_percentages[0] if len(urban_percentages) >= 2 else 0,
                'average_annual_growth': sum(growth_rates) / len(growth_rates) if growth_rates else 0,
                'peak_growth_year': years[growth_rates.index(max(growth_rates)) + 1] if growth_rates else None,
                'peak_growth_rate': max(growth_rates) if growth_rates else 0
            }
        
        return temporal_data
    
    def save_batch_results(self, results: dict, visualize: bool = True) -> str:
        """Save all batch processing results."""
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = self.output_dir / f"batch_processing_{timestamp}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {batch_output_dir}")
        
        # Save individual results
        for year, result in results.items():
            result_dir = batch_output_dir / f"year_{year}"
            result.save_results(str(result_dir), save_models=False)
        
        # Generate and save batch summary
        summary = self.generate_batch_summary(results)
        summary_path = batch_output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate and save temporal analysis
        temporal_analysis = self.create_temporal_analysis(results)
        temporal_path = batch_output_dir / "temporal_analysis.json"
        with open(temporal_path, 'w') as f:
            json.dump(temporal_analysis, f, indent=2)
        
        # Create visualizations if requested
        if visualize:
            logger.info("Generating batch visualizations...")
            
            # Create simple batch visualization
            self._create_simple_batch_visualization(results, batch_output_dir)
        
        # Save processing log
        log_data = {
            'input_directory': str(self.input_dir),
            'output_directory': str(batch_output_dir),
            'processing_timestamp': datetime.now().isoformat(),
            'total_images_processed': len(results),
            'years_processed': list(results.keys()),
            'pipeline_configuration': {
                'n_clusters': [2],
                'n_pca_components': 8,
                'svm_kernel': 'linear',
                'binary_classification': True
            }
        }
        
        log_path = batch_output_dir / "processing_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Batch processing completed successfully!")
        logger.info(f"Results saved to: {batch_output_dir}")
        
        return str(batch_output_dir)
    
    def _save_batch_summary(self, results: dict, batch_output_dir: Path) -> None:
        """Save batch summary and visualizations."""
        logger.info("Generating batch summary...")
        
        try:
            # Generate and save batch summary
            summary = self.generate_batch_summary(results)
            summary_path = batch_output_dir / "batch_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate and save temporal analysis
            temporal_analysis = self.create_temporal_analysis(results)
            temporal_path = batch_output_dir / "temporal_analysis.json"
            with open(temporal_path, 'w') as f:
                json.dump(temporal_analysis, f, indent=2)
            
            # Create visualization
            self._create_simple_batch_visualization(results, batch_output_dir)
            
            # Save processing log
            log_data = {
                'input_directory': str(self.input_dir),
                'output_directory': str(batch_output_dir),
                'processing_timestamp': datetime.now().isoformat(),
                'total_images_processed': len(results),
                'years_processed': list(results.keys()),
                'pipeline_configuration': {
                    'n_clusters': [2],
                    'n_pca_components': 8,
                    'svm_kernel': 'linear',
                    'binary_classification': True
                }
            }
            
            log_path = batch_output_dir / "processing_log.json"
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"✅ Batch summary saved to: {batch_output_dir}")
            
        except Exception as e:
            logger.warning(f"Could not save batch summary: {e}")
    
    def _create_simple_comparison(self, result, year: str, image_path: str, output_dir: Path) -> None:
        """Create simple comparison: original vs binary classification."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image
            
            # Load original image
            original_image = Image.open(image_path)
            original_array = np.array(original_image)
            
            # Get binary classification (SVM with 2 classes)
            if hasattr(result, 'prediction_maps') and 'svm_2_classes' in result.prediction_maps:
                binary_pred = result.prediction_maps['svm_2_classes'].copy()
                
                # Smart cluster labeling using multiple heuristics
                self._standardize_cluster_labels(binary_pred, original_array, year)
                
                # Create figure with 2 subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                axes[0].imshow(original_array)
                axes[0].set_title(f'Imagen Original {year}', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # Binary classification with custom colors: green=non-urban(0), red=urban(1)
                from matplotlib.colors import ListedColormap
                colors = ['#2E8B57', '#DC143C']  # Green for non-urban (0), Red for urban (1)
                custom_cmap = ListedColormap(colors)
                
                im = axes[1].imshow(binary_pred, cmap=custom_cmap, vmin=0, vmax=1)
                axes[1].set_title(f'Clasificación Binaria {year}\n(Verde=No-urbano, Rojo=Urbano)', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                # Add colorbar with custom labels
                cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
                cbar.set_label('Tipo de Cobertura', rotation=270, labelpad=15)
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(['No-urbano\n(Verde)', 'Urbano\n(Rojo)'])
                
                # Calculate urban area statistics
                urban_pixels = int(np.sum(binary_pred))
                total_pixels = binary_pred.size
                urban_percentage = (urban_pixels / total_pixels) * 100
                
                # Add overall title with statistics
                fig.suptitle(f'Análisis de Crecimiento Urbano - {year}\n'
                           f'Área urbana: {urban_percentage:.1f}% ({urban_pixels:,} píxeles de {total_pixels:,})',
                           fontsize=16, fontweight='bold', y=0.95)
                
                plt.tight_layout()
                
                # Save the comparison
                viz_path = output_dir / f"comparison_{year}.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()  # Important: close to free memory
                
                logger.info(f"✅ Visualization saved: {viz_path}")
                
            else:
                logger.warning(f"⚠️ No binary prediction found for {year}")
                
        except Exception as e:
            logger.error(f"❌ Error creating visualization for {year}: {e}")
            # Ensure figure is closed even on error
            try:
                plt.close()
            except:
                pass
    
    def _standardize_cluster_labels(self, binary_pred, original_array, year: str) -> None:
        """Standardize cluster labels so that 1=urban (red), 0=non-urban (green)."""
        try:
            import numpy as np
            
            # Convert to grayscale for brightness analysis
            if len(original_array.shape) == 3:
                original_gray = np.mean(original_array, axis=2)
            else:
                original_gray = original_array
            
            # Calculate approximate NDVI using RGB bands
            if len(original_array.shape) == 3 and original_array.shape[2] >= 3:
                R = original_array[:, :, 0].astype(float)
                G = original_array[:, :, 1].astype(float) 
                B = original_array[:, :, 2].astype(float)
                
                # Approximate NDVI using NIR ≈ G and Red ≈ R
                # NDVI = (NIR - Red) / (NIR + Red)
                ndvi_approx = np.divide(G - R, G + R + 1e-8)  # Add small epsilon to avoid division by zero
                
                # Calculate mean NDVI for each cluster
                cluster_0_mask = binary_pred == 0
                cluster_1_mask = binary_pred == 1
                
                if np.sum(cluster_0_mask) > 100 and np.sum(cluster_1_mask) > 100:
                    mean_ndvi_0 = np.mean(ndvi_approx[cluster_0_mask])
                    mean_ndvi_1 = np.mean(ndvi_approx[cluster_1_mask])
                    
                    # Urban areas have lower NDVI (less vegetation)
                    # We want urban to be 1 (red), so if cluster 0 has lower NDVI, flip labels
                    if mean_ndvi_0 < mean_ndvi_1:
                        binary_pred[:] = 1 - binary_pred  # Flip in-place
                        logger.info(f"🔄 NDVI-based flip for {year}: NDVI_0={mean_ndvi_0:.3f} < NDVI_1={mean_ndvi_1:.3f}")
                    else:
                        logger.info(f"✅ NDVI-based labels OK for {year}: NDVI_1={mean_ndvi_1:.3f} <= NDVI_0={mean_ndvi_0:.3f}")
                    return
            
            # Fallback: brightness-based heuristic
            cluster_0_mask = binary_pred == 0
            cluster_1_mask = binary_pred == 1
            
            if np.sum(cluster_0_mask) > 100 and np.sum(cluster_1_mask) > 100:
                mean_brightness_0 = np.mean(original_gray[cluster_0_mask])
                mean_brightness_1 = np.mean(original_gray[cluster_1_mask])
                
                # Urban areas tend to be brighter (concrete, buildings)
                # We want urban to be 1 (red), so if cluster 0 is brighter, flip labels
                if mean_brightness_0 > mean_brightness_1:
                    binary_pred[:] = 1 - binary_pred  # Flip in-place
                    logger.info(f"🔄 Brightness-based flip for {year}: Bright_0={mean_brightness_0:.1f} > Bright_1={mean_brightness_1:.1f}")
                else:
                    logger.info(f"✅ Brightness-based labels OK for {year}: Bright_1={mean_brightness_1:.1f} >= Bright_0={mean_brightness_0:.1f}")
            else:
                logger.warning(f"⚠️ Could not analyze clusters for {year}, using original labels")
                
        except Exception as e:
            logger.warning(f"⚠️ Error in label standardization for {year}: {e}")
    
    def _create_simple_batch_visualization(self, results: dict, output_dir: Path) -> None:
        """Create simple batch visualization showing urban growth over time."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
            
            # Extract years and urban percentages from saved NPY files
            years = []
            urban_percentages = []
            
            # Look for year directories in the output folder
            base_dir = output_dir.parent if output_dir.name == 'visualizations' else output_dir
            year_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('year_')]
            
            for year_dir in sorted(year_dirs):
                try:
                    year = year_dir.name.replace('year_', '')
                    svm_file = year_dir / 'prediction_maps' / 'svm_2_classes.npy'
                    
                    if svm_file.exists():
                        binary_pred = np.load(svm_file)
                        
                        # Calculate urban percentage (assuming 1 = urban after standardization)
                        urban_pixels = np.sum(binary_pred == 1)
                        total_pixels = binary_pred.size
                        urban_pct = (urban_pixels / total_pixels) * 100
                        
                        years.append(int(year))
                        urban_percentages.append(float(urban_pct))
                        
                        logger.info(f"📊 {year}: {urban_pct:.1f}% urban ({urban_pixels:,} / {total_pixels:,} pixels)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not process year {year_dir.name}: {e}")
                    continue
            
            if len(years) > 1:
                # Create temporal evolution plot
                plt.figure(figsize=(14, 8))
                
                # Main plot with trend line
                plt.plot(years, urban_percentages, 'b-o', linewidth=3, markersize=8, label='Urban Area %')
                
                # Add trend line
                if len(years) > 2:
                    z = np.polyfit(years, urban_percentages, 1)
                    p = np.poly1d(z)
                    plt.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2f}%/year')
                
                plt.title('Evolución del Crecimiento Urbano de Querétaro (1984-2020)', 
                         fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('Año', fontsize=14, fontweight='bold')
                plt.ylabel('Área Urbana (%)', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)
                
                # Add statistics annotation
                if len(urban_percentages) >= 2:
                    initial_urban = urban_percentages[0]
                    final_urban = urban_percentages[-1]
                    growth = final_urban - initial_urban
                    years_span = years[-1] - years[0]
                    
                    stats_text = (f'Crecimiento total: +{growth:.1f}%\n'
                                f'Periodo: {years_span} años\n'
                                f'Tasa promedio: {growth/years_span:.2f}%/año')
                    
                    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                
                # Save the temporal plot
                viz_path = output_dir / "urban_growth_timeline.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"✅ Temporal visualization saved: {viz_path}")
                logger.info(f"📈 Analyzed {len(years)} years from {min(years)} to {max(years)}")
                
            else:
                logger.warning(f"⚠️ Not enough data points for temporal visualization ({len(years)} years)")
        
        except Exception as e:
            logger.error(f"❌ Error creating batch visualization: {e}")
            # Ensure figure is closed even on error
            try:
                plt.close()
            except:
                pass


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Satellite Image Processing Pipeline CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images in batch')
    batch_parser.add_argument('input_dir', help='Directory containing input images (PNG files)')
    batch_parser.add_argument('--output-dir', default='data/processed', 
                             help='Output directory for results (default: data/processed)')
    batch_parser.add_argument('--sample-size', type=int, default=10000,
                             help='Sample size for training (default: 10000)')
    batch_parser.add_argument('--visualize', action='store_true',
                             help='Generate visualization plots')
    
    # Single image command
    single_parser = subparsers.add_parser('single', help='Process a single image')
    single_parser.add_argument('image_path', help='Path to input image')
    single_parser.add_argument('--output-dir', default='data/processed',
                              help='Output directory for results')
    single_parser.add_argument('--sample-size', type=int, default=10000,
                              help='Sample size for training (default: 10000)')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        # Batch processing
        processor = BatchProcessor(args.input_dir, args.output_dir)
        results = processor.process_all_images(sample_size=args.sample_size)
        
        if results:
            print(f"\n✅ Batch processing completed!")
            print(f"📁 Results saved to: {processor.last_output_dir}")
            print(f"🎯 Processed {len(results)} images with binary classification")
            print(f"📊 Each image saved individually during processing")
        else:
            print("❌ No images were processed successfully")
            sys.exit(1)
    
    elif args.command == 'single':
        # Single image processing
        pipeline = SatelliteImagePipeline(
            n_clusters_list=[2],  # Binary classification
            n_pca_components=8,
            extract_full_features=True,
            svm_kernel='linear',
            random_state=42
        )
        
        result = pipeline.process_image(args.image_path, sample_size=args.sample_size)
        output_path = result.save_results(args.output_dir, save_models=True)
        
        print(f"\n✅ Image processing completed!")
        print(f"📁 Results saved to: {output_path}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()