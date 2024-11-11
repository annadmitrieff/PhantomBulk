#!/usr/bin/env python3
# post_process.py

import argparse
import logging
import sys
from pathlib import Path
import shutil
import subprocess
import re

from config import Config  # Adjust the import path as necessary

def setup_logging(log_level: str):
    """Set up logging for the script."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("post_process.log")
        ]
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-process PHANTOM simulations by renaming dump files and submitting MCFOST jobs."
    )
    parser.add_argument(
        '-c', '--config_file',
        type=str,
        default='$HOME/PhantomBulk/config/config.yaml',
        help='Path to configuration file (YAML format, default: $HOME/PhantomBulk/config/config.yaml)'
    )
    parser.add_argument(
        '-p', '--sim_path',
        type=str,
        required=True,
        help='Directory containing simulation runs (e.g., sim_*/).'
    )
    # Output directory is now taken from config.yaml, no need to specify via command line
    return parser.parse_args()

def main():
    """Main function to execute the post-processing."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config_path = Path(args.config_file).expanduser()
    if not config_path.is_file():
        print(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    
    config = Config(str(config_path))
    
    # Set up logging
    setup_logging(config.log_level)
    
    # Assign variables from config
    sim_path = Path(args.sim_path).expanduser()
    target_dir = config.OUTPUT_DIR
    ref_file = config.REFERENCE_FILE
    mcfost_exec = config.MCFOST_EXEC
    ld_linux = config.LD_LINUX
    
    # Validate simulation path
    if not sim_path.is_dir():
        logging.error(f"Simulation path '{sim_path}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Validate reference file
    if not ref_file.is_file():
        logging.error(f"Reference file '{ref_file}' does not exist.")
        sys.exit(1)
    
    # Validate MCFOST executable
    if not mcfost_exec.is_file():
        logging.error(f"MCFOST executable '{mcfost_exec}' does not exist.")
        sys.exit(1)
    
    # Validate LD_LINUX
    if not ld_linux.is_file():
        logging.error(f"LD Linux loader '{ld_linux}' does not exist.")
        sys.exit(1)
    
    logging.info(f"Looking for simulations in '{sim_path}'...")
    logging.info(f"Using MCFOST parameter file '{ref_file}'.")
    logging.info(f"Post-processed outputs will be saved in '{target_dir}'.")
    
    # Create the target directory if it doesn't exist
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured target directory '{target_dir}' exists.")
    except Exception as e:
        logging.error(f"Failed to create target directory '{target_dir}': {e}")
        sys.exit(1)
    
    logging.info("Starting post-processing of simulations...")
    
    # Initialize a counter for processed simulations
    processed_count = 0
    
    # Iterate through each simulation directory
    for sim_dir in sim_path.iterdir():
        if sim_dir.is_dir():
            setup_file = sim_dir / 'dustysgdisc.setup'
            dump_file = sim_dir / 'dustysgdisc_00020'  # Adjust as needed
            
            if setup_file.is_file():
                if dump_file.is_file():
                    sim_name = sim_dir.name
                    dest_dir = target_dir / sim_name
                    job_name = dest_dir / f"{sim_name}-mcfost.sh"
                    
                    # Create destination directory
                    try:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        logging.info(f"Created directory '{dest_dir}'.")
                    except Exception as e:
                        logging.error(f"Failed to create directory '{dest_dir}': {e}")
                        continue
                    
                    # Copy necessary files
                    try:
                        shutil.copy2(dump_file, dest_dir)
                        shutil.copy2(setup_file, dest_dir)
                        shutil.copy2(ref_file, dest_dir)
                        logging.info(f"Copied files to '{dest_dir}'.")
                    except Exception as e:
                        logging.error(f"Failed to copy files to '{dest_dir}': {e}")
                        continue
                    
                    # Create the SLURM submission script
                    try:
                        job_script_content = f"""#!/bin/bash
#SBATCH --job-name={sim_name}-mcfost                # Job name
#SBATCH --partition={config.PARTITION}              # Partition name (batch, highmem_p, or gpu_p)
#SBATCH --ntasks={config.N_TASKS}                   # Number of tasks
#SBATCH --cpus-per-task={config.CPUS_PER_TASK}      # CPU core count per task
#SBATCH --mem={config.MEM}                          # Memory per node
#SBATCH --time={config.TIME}                        # Time limit (days-hours:minutes:seconds)
#SBATCH --output={dest_dir}/%x_%j.out               # Standard output log
#SBATCH --mail-user={config.USER_EMAIL}             # User email
#SBATCH --mail-type={config.MAIL_TYPE}              # Mail events (BEGIN, END, FAIL, ALL)
    
# Change to the simulation directory
cd {dest_dir}
    
# Create fits directory
mkdir -p {target_dir}/fits
    
# EXECUTING MCFOST COMMANDS
{ld_linux} --library-path "{ld_linux.parent}:/lib64" {mcfost_exec} {ref_file.name} -phantom dustysgdisc_00020 -fix_star -mol
    
{ld_linux} --library-path "{ld_linux.parent}:/lib64" {mcfost_exec} {ref_file.name} -phantom dustysgdisc_00020 -fix_star -img 1300
    
echo "Processing completed. Moving files:"
    
DATA_DIR="{dest_dir}/data_1300"
    
echo "Looking for .fits.gz files in $DATA_DIR..."
    
gunzip $DATA_DIR/RT.fits.gz
    
echo "Unzipped .fits.gz file $DATA_DIR/RT.fits.gz..."
    
mv $DATA_DIR/RT.fits {target_dir}/fits/{sim_name}.fits
    
echo "Moved $DATA_DIR/RT.fits to {target_dir}/fits/{sim_name}.fits."
"""
                        with open(job_name, 'w') as job_file:
                            job_file.write(job_script_content)
                        job_name.chmod(job_name.stat().st_mode | 0o111)
                        logging.info(f"Created SLURM job script '{job_name}'.")
                    except Exception as e:
                        logging.error(f"Failed to create SLURM job script '{job_name}': {e}")
                        continue
                    
                    # Submit the job
                    try:
                        subprocess.run(['sbatch', str(job_name)], check=True)
                        logging.info(f"Submitted job '{job_name}'.")
                        processed_count += 1
                    except subprocess.CalledProcessError as e:
                        logging.error(f"Failed to submit job '{job_name}': {e}")
                        continue
                else:
                    logging.warning(f"No dump file found in '{sim_dir}'. Skipping.")
            else:
                logging.warning(f"No setup file found in '{sim_dir}'. Skipping.")
    
    logging.info(f"Post-processing complete. {processed_count} simulations processed and jobs submitted.")

if __name__ == "__main__":
    main()
