#!/usr/bin/env python3
# grab_fits.py
# Python script to rename and copy RT.fits files from simulation directories.

import os
import re
import shutil
import argparse
from pathlib import Path
import logging
import sys

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("grab_fits.log")
        ]
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rename and copy RT.fits files from simulation directories."
    )
    parser.add_argument(
        '-d', '--destination',
        type=str,
        default='fitsfiles',
        help='Destination directory for copied FITS files (default: fitsfiles)'
    )
    return parser.parse_args()

def main():
    """Main function to execute the script."""
    setup_logging()
    args = parse_arguments()
    destination = Path(args.destination).expanduser()
    
    # Create destination directory if it doesn't exist
    try:
        destination.mkdir(parents=True, exist_ok=True)
        logging.info(f"Destination directory set to '{destination}'.")
    except Exception as e:
        logging.error(f"Failed to create destination directory '{destination}': {e}")
        sys.exit(1)
    
    # Initialize a counter for processed files
    count = 0
    
    # Use glob to find all RT.fits files in sim_*/data_1300/ directories
    sim_dirs = Path('.').glob('sim_*/data_1300/RT.fits')
    
    for filepath in sim_dirs:
        try:
            # Extract the simulation number using regex
            match = re.match(r'sim_0*([0-9]+)/data_1300/RT\.fits', str(filepath))
            if not match:
                logging.warning(f"Could not extract simulation number from '{filepath}'. Skipping.")
                continue
            sim_num = match.group(1)
            
            # Define new filename and path
            new_filename = f"{sim_num}.fits"
            new_filepath = filepath.parent / new_filename
            
            # Rename the file in place
            filepath.rename(new_filepath)
            logging.info(f"Renamed '{filepath}' to '{new_filepath}'.")
            
            # Copy the renamed file to the destination directory
            shutil.copy2(new_filepath, destination)
            logging.info(f"Copied '{new_filepath}' to '{destination}/'.")
            
            count += 1
        except Exception as e:
            logging.error(f"Error processing file '{filepath}': {e}")
            continue
    
    logging.info(f"Processing complete. {count} files have been renamed and copied to '{destination}/'.")

if __name__ == "__main__":
    main()
