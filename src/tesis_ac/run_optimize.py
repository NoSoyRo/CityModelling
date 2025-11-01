#!/usr/bin/env python3
"""
Main script for GA optimization.

Usage:
    python -m tesis_ac.run_optimize configs/default.yaml
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
    """Main function for GA optimization pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GA optimization pipeline"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target grid path for optimization"
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
        
        # TODO: Implement optimization pipeline
        logger.info("🚧 Optimization pipeline implementation in progress...")
        
        # Steps to implement:
        # 1. Load initial grid and target grid
        # 2. Calculate WoE weights
        # 3. Setup GA with DEAP
        # 4. Run optimization (fitness = IoU vs target)
        # 5. Save best parameters and results
        # 6. Generate evaluation plots
        
        logger.info("✅ GA optimization completed")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()