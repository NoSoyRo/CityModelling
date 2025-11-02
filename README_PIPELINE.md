# Satellite Image Classification Pipeline

A complete pipeline for processing satellite imagery using machine learning techniques including feature extraction, clustering, and classification.

## Features

- **Feature Extraction**: RGB channels, texture features (LBP, Sobel), spectral indices (NDVI approximation)
- **Preprocessing**: Normalization and PCA dimensionality reduction
- **Clustering**: K-means clustering with multiple cluster numbers
- **Classification**: SVM classification using clustering results as ground truth
- **Visualization**: Comprehensive plotting and analysis tools
- **Batch Processing**: Process multiple images with temporal analysis
- **Modular Design**: Easy to extend and customize

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CityModelling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Process a single image:
```bash
python cli.py single data/raw/imagen_2020.png --visualize --save-models
```

Process multiple images:
```bash
python cli.py batch data/raw --visualize
```

Analyze existing results:
```bash
python cli.py analyze data/processed/satellite_classification_20231101_143022 --visualize
```

### Python API

```python
from src.tesis_ac.pipeline.main_pipeline import SatelliteImagePipeline

# Create pipeline
pipeline = SatelliteImagePipeline(
    n_clusters_list=[2, 3, 4],
    n_pca_components=8,
    extract_full_features=True
)

# Process image
pipeline.process_image("data/raw/imagen_2020.png")

# Get results
summary = pipeline.get_summary()
prediction_maps = pipeline.prediction_maps
metrics = pipeline.metrics

# Save results
output_dir = pipeline.save_results("output", save_models=True)
```

## Pipeline Architecture

### 1. Feature Extraction (`feature_extraction.py`)

- **FeatureExtractor**: Extracts features from RGB satellite images
  - RGB channel statistics
  - Texture features (LBP, Sobel gradients, entropy)
  - Spectral indices (NDVI approximation, excess green/red)
  - Color space transformations (LAB, HSV)

- **FeaturePreprocessor**: Normalizes and reduces dimensionality
  - StandardScaler normalization
  - PCA dimensionality reduction
  - Variance analysis

### 2. Clustering and Classification (`clustering.py`)

- **ImageClusterer**: K-means clustering for image segmentation
  - Multiple cluster numbers (2, 3, 4)
  - Automatic best clustering selection
  - Binary mask generation

- **ImageClassifier**: SVM classification
  - Uses clustering results as ground truth
  - Multiple kernel options
  - Performance evaluation

- **SatelliteImageProcessor**: Complete ML pipeline
  - Orchestrates clustering and classification
  - Provides unified interface
  - Metrics collection

### 3. Main Pipeline (`main_pipeline.py`)

- **SatelliteImagePipeline**: End-to-end processing
  - Single image processing
  - Batch processing for multiple images
  - Results saving and loading
  - Model persistence

### 4. Visualization (`visualization.py`)

- Single image result visualization
- Metrics comparison plots
- Temporal analysis for multiple years
- Prediction map grids
- Feature importance analysis

## Output Structure

```
data/processed/satellite_classification_YYYYMMDD_HHMMSS/
├── metadata.json                 # Processing parameters and metrics
├── prediction_maps/             # Numpy arrays of prediction maps
│   ├── kmeans_2_clusters.npy
│   ├── kmeans_3_clusters.npy
│   ├── svm_2_classes.npy
│   └── ...
├── models/                      # Trained models (if saved)
│   ├── preprocessor.pkl
│   ├── clustering_models.pkl
│   └── classification_models.pkl
├── features/                    # Feature matrices (if saved)
│   ├── X_features.npy
│   └── X_pca.npy
└── *.png                       # Visualization plots
```

## Configuration Options

### Pipeline Parameters

- `n_clusters_list`: List of cluster numbers to try (default: [2, 3, 4])
- `n_pca_components`: Number of PCA components (default: 8)
- `extract_full_features`: Extract full feature set vs basic (default: True)
- `svm_kernel`: SVM kernel type ('linear', 'rbf', 'poly') (default: 'linear')
- `random_state`: Random seed for reproducibility (default: 42)

### Processing Parameters

- `sample_size`: Maximum samples for SVM training (default: 5000)
- `test_size`: Fraction for testing in SVM (default: 0.2)

## Examples

See `examples/satellite_classification_example.py` for complete examples:

1. **Single Image Processing**: Load, process, and visualize one image
2. **Batch Processing**: Process multiple images with temporal analysis
3. **Loading Results**: Load and analyze previously saved results

## Performance Metrics

The pipeline provides comprehensive metrics:

### Clustering Metrics
- **Inertia**: Within-cluster sum of squared distances
- **Distribution**: Number of pixels per cluster

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **Training samples**: Number of samples used

### Feature Metrics
- **PCA Variance**: Variance explained by components
- **Feature Importance**: Relative importance of features

## Extending the Pipeline

### Adding New Features

```python
# In feature_extraction.py
def extract_custom_features(self, image):
    # Add your feature extraction logic
    custom_features = {
        'my_feature': calculate_my_feature(image)
    }
    return custom_features
```

### Adding New Algorithms

```python
# In clustering.py
from sklearn.cluster import DBSCAN

class CustomClusterer:
    def __init__(self):
        self.model = DBSCAN()
    
    def fit_predict(self, X):
        return self.model.fit_predict(X)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `sample_size` or use `extract_full_features=False`
2. **Long Processing Times**: Use smaller images or reduce PCA components
3. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

- Use `extract_full_features=False` for faster processing
- Reduce `sample_size` for faster SVM training
- Use fewer PCA components for speed vs quality tradeoff
- Process images in smaller batches for memory efficiency

## Dependencies

- NumPy: Numerical computations
- OpenCV: Image processing
- scikit-learn: Machine learning algorithms
- scikit-image: Advanced image processing
- matplotlib: Visualization
- pandas: Data analysis (optional)

## License

[Add your license information here]

## Citation

[Add citation information if this is for academic use]