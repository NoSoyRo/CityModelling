#!/usr/bin/env python3
"""
Main script for classification and grid generation.

Usage:
    python -m tesis_ac.run_classify_and_grid configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from tesis_ac.config import load_config, create_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for classification and grid generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run classification and grid generation pipeline"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input image path (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Create directories
        create_directories(config)
        logger.info("Created output directories")
        
        # TODO: Implement pipeline steps
        logger.info("🚧 Pipeline implementation in progress...")
        
        # Steps to implement:
        # 1. Load input image
        # 2. Extract features (LBP, Sobel, RGB)
        # 3. Cluster features (KMeans/GMM)
        # 4. Assign semantic labels
        # 5. Convert to grid
        # 6. Save results
        
        logger.info("✅ Classification and grid generation completed")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()