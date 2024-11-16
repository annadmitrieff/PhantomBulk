#!/usr/bin/env python3
# src/PhantomBulk/main.py

import argparse
import logging
import yaml
import pandas as pd
import sys
from pathlib import Path
import subprocess  # Needed for subprocess operations
from dataclasses import dataclass, field, asdict

from PhantomBulk.config import Config
from PhantomBulk.generators import PhysicalPPDGenerator, PPDParameters
from PhantomBulk.file_manager import PHANTOMFileManager

def generate_phantom_input(
    params: PPDParameters,
    output_dir: Path,
    sim_id: int,
    file_manager: PHANTOMFileManager,
    config: Config
) -> bool:

    """
    Generate PHANTOM setup file, run phantomsetup, modify .in, and create submission script.

    Parameters:
        params (PPDParameters): The PPD parameters.
        output_dir (Path): Directory to store simulation files.
        sim_id (int): Simulation identifier.
        file_manager (PHANTOMFileManager): File manager instance.
        config (Config): Configuration instance.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Step 1: Locate `phantom` and `phantomsetup` executables
    try:
        phantom_exe, phantomsetup_exe = file_manager.phantom_executables
    except AttributeError:
        logging.error("PHANTOM executables not found.")
        return False

    # Step 2: Generate the populated `.setup` file
    try:
        file_manager.create_setup_file(params, output_dir, sim_id)
        logging.info(f"Generated setup file for simulation {sim_id} in '{output_dir}'.")
    except Exception as e:
        logging.error(f"Failed to create setup file for simulation {sim_id}: {e}")
        return False

    # Step 3: Ensure executables are executable
    for exe in [phantom_exe, phantomsetup_exe]:
        try:
            exe.chmod(exe.stat().st_mode | 0o111)
            logging.debug(f"Set executable permissions for '{exe}'.")
        except Exception as e:
            logging.error(f"Failed to set executable permissions for '{exe}': {e}")
            return False

    # Step 4: Run `phantomsetup` to generate `dustysgdisc.in`
    try:
        # Run phantomsetup twice to add add'l values (e.g. oblateness calculations) that phantom auto-calculates
        subprocess.run([str(phantomsetup_exe), 'dustysgdisc'], cwd=output_dir, check=True)
        subprocess.run([str(phantomsetup_exe), 'dustysgdisc'], cwd=output_dir, check=True)
        logging.info(f"Ran 'phantomsetup' for simulation {sim_id}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"'phantomsetup' failed for simulation {sim_id}: {e}")
        return False

    # Step 5: Modify `.in` file based on additional parameters
    try:
        file_manager.modify_in_file(params, output_dir)
        logging.info(f"Modified '.in' file for simulation {sim_id}.")
    except FileNotFoundError as e:
        logging.error(f"Failed to modify '.in' file for simulation {sim_id}: {e}")
        return False

    # Step 6: Generate the submission script
    submission_script_content = f"""#!/bin/bash
#SBATCH --job-name=ppd_{sim_id:04d}                                # Job name
#SBATCH --partition={config.PARTITION}                             # Partition name
#SBATCH --ntasks={config.N_TASKS}                                  # Number of tasks
#SBATCH --cpus-per-task={config.CPUS_PER_TASK}                    # CPU core count per task
#SBATCH --mem={config.MEM}                                         # Memory per node
#SBATCH --time={config.TIME}                                       # Time limit (days-hours:minutes:seconds)
#SBATCH --output={output_dir}/ppd_{sim_id:04d}_%j.out              # Standard output log
#SBATCH --mail-user={config.USER_EMAIL}                            # User email
#SBATCH --mail-type={config.MAIL_TYPE}                            # Mail events (BEGIN, END, FAIL, ALL)

# PHANTOM contingencies
export SYSTEM=gfortran
ulimit -s unlimited
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=28
export OMP_STACKSIZE=1024M

# Change to the simulation directory
cd {output_dir}

# Run PHANTOM with the input file
{phantom_exe} dustysgdisc.in
"""

    # Write the submission script
    submit_script_path = output_dir / f'run_{sim_id:04d}.sh'
    try:
        with open(submit_script_path, 'w') as f:
            f.write(submission_script_content)
        submit_script_path.chmod(submit_script_path.stat().st_mode | 0o111)
        logging.debug(f"Created submission script '{submit_script_path}'.")
    except Exception as e:
        logging.error(f"Failed to create submission script for simulation {sim_id}: {e}")
        return False

    logging.info(f"Submission script created for simulation {sim_id}.")

    return True

def main():
    """
    Main function to generate PPD simulations.
    """
    # Configure logging
    logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phantombulk_debug.log"),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ])

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Protoplanetary Disc Simulations")

    # Positional argument for number of simulations
    parser.add_argument('-n', '--n_sims', type=int, help='Number of simulations to generate')

    # Optional arguments for flexibility
    parser.add_argument('-d', '--output_dir', type=str, default='$HOME/PhantomBulk/outputs/',
                        help='Output directory for simulations (default: PhantomBulk/outputs/)')
    parser.add_argument('-c', '--config_file', type=str, default='$HOME/PhantomBulk/config/config.yaml',
                        help='Path to configuration file (YAML format, default: PhantomBulk/config/config.yaml)')

    args = parser.parse_args()

    # Load configuration from file
    config_path = Path(args.config_file).expanduser()
    if config_path.is_file():
        config = Config(str(config_path))
        logging.info(f"Loaded configuration from '{config_path}'.")
    else:
        logging.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)

    # Assign arguments to variables
    n_sims = args.n_sims
    output_dir = Path(args.output_dir).expanduser()

    logging.info(f"Number of simulations to generate: {n_sims}")
    logging.info(f"Output directory: {output_dir}")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator and file manager
    generator = PhysicalPPDGenerator(config)
    file_manager = PHANTOMFileManager(config)

    # Create a submit_all.sh script
    submit_all_path = output_dir / 'submit_all.sh'
    try:
        with open(submit_all_path, 'w') as f:
            f.write('#!/bin/bash\n')
        submit_all_path.chmod(submit_all_path.stat().st_mode | 0o111)
        logging.info(f"Created submission script '{submit_all_path}'.")
    except Exception as e:
        logging.error(f"Failed to create 'submit_all.sh': {e}")
        sys.exit(1)

    # Generate parameters and setup/input files
    param_records = []
    for i in range(n_sims):
        # Generate physically consistent parameters
        try:
            params = generator.generate_single_ppd()
            logging.debug(f"Generated parameters for simulation {i}.")
        except ValueError as e:
            logging.warning(f"Simulation {i}: {e}. Skipping.")
            continue

        # Set up simulation directory
        sim_dir = output_dir / f'sim_{i:04d}'
        try:
            sim_dir.mkdir(exist_ok=True)
            logging.debug(f"Created simulation directory '{sim_dir}'.")
        except Exception as e:
            logging.error(f"Failed to create simulation directory '{sim_dir}': {e}")
            continue

        # Generate .setup file, run phantomsetup, modify .in, and create submission script
        success = generate_phantom_input(params, sim_dir, i, file_manager, config)

        if not success:
            logging.warning(f"Simulation {i}: Failed to generate input files. Skipping.")
            continue

        # Add job to submit_all.sh
        try:
            with open(submit_all_path, 'a') as f:
                f.write(f'sbatch {sim_dir}/run_{i:04d}.sh\n')
            logging.debug(f"Added job submission for simulation {i} to 'submit_all.sh'.")
        except Exception as e:
            logging.error(f"Failed to add job submission for simulation {i}: {e}")
            continue

        # Record parameters in a dictionary
        param_dict = asdict(params)
        param_dict['simulation_id'] = i
        # Convert planets to list of dictionaries
        param_dict['n_planets'] = len(params.planets)
        param_dict['planets'] = [asdict(planet) for planet in params.planets] if params.planets else []
        param_records.append(param_dict)

        # Optional: Print progress
        if (i + 1) % 100 == 0 or (i + 1) == n_sims:
            logging.info(f"Generated {i + 1}/{n_sims} simulations")

    # Save parameters to CSV
    if param_records:
        df = pd.DataFrame(param_records)
        param_db_path = output_dir / 'input_parameters.csv'
        try:
            df.to_csv(param_db_path, index=False)
            logging.info(f"Saved simulation parameters to '{param_db_path}'.")
        except Exception as e:
            logging.error(f"Failed to save simulation parameters to CSV: {e}")
    else:
        logging.warning("No simulation parameters to save.")

    # Summary output
    logging.info(f"\nGenerated {len(param_records)} disc configurations")
    logging.info(f"Files saved in: {output_dir}")

    # ===================================
    # ++ INTERACTIVE SUBMISSION PROMPT ++
    # ===================================

    # Define submission command based on scheduler
    scheduler = config.JOB_SCHEDULER.upper()
    job_scheduler_map = {
        'SLURM': 'sbatch',
        'PBS': 'qsub',
        'SGE': 'qsub -cwd'
    }

    if scheduler not in job_scheduler_map:
        logging.error(f"Unsupported job scheduler: {scheduler}")
        sys.exit(1)

    SUBMIT_CMD = job_scheduler_map[scheduler]
    logging.info(f"Using job scheduler: {scheduler}")

    # Check if submit_all.sh exists
    if not submit_all_path.is_file():
        logging.error(f"Submission script '{submit_all_path}' not found.")
        sys.exit(1)

    # Prompt the user to decide whether to submit all jobs (when uncommented)
    echo_output = f"""
================================================================================================
You have generated {n_sims} simulations in '{output_dir}'.
It's recommended to verify the '.setup' and '.in' files before submission.
To submit all, navigate to {output_dir} and submit the script 'submit_all.sh' to your scheduler.
================================================================================================
"""

#Excluded: currently non-functional.
'''
    print(echo_output)

    while True:
        yn = input("Do you want to execute 'submit_all.sh' and submit all jobs? [y/n]: ").strip().lower()
        if yn in ['y', 'yes']:
            print(f"\nSubmitting all jobs in '{output_dir}' using scheduler '{scheduler}'...")
            try:
                with open(submit_all_path, 'r') as f:
                    for job_script in f:
                        job_script = job_script.strip()
                        job_script_path = Path(job_script)
                        if job_script_path.is_file():
                            try:
                                # Split SUBMIT_CMD in case it contains additional flags like '-cwd'
                                cmd_parts = SUBMIT_CMD.split() + [str(job_script_path)]
                                result = subprocess.run(cmd_parts, check=True)
                                print(f"Submitted job: {job_script}")
                            except subprocess.CalledProcessError:
                                print(f"Error: Job submission failed for script '{job_script}'.")
                        else:
                            print(f"Warning: Job script '{job_script}' does not exist. Skipping.")
                print("All simulations have been submitted successfully.")
            except Exception as e:
                logging.error(f"Failed to submit jobs: {e}")
            break
        elif yn in ['n', 'no', '']:
            print(f"\nJob submission skipped. You can verify the '.setup' and '.in' files in '{output_dir}' before submitting manually.")
            print(f"To submit all jobs later, execute the 'submit_all.sh' script in '{output_dir}'.")
            break
        else:
            print("Please answer yes (y) or no (n).")

    sys.exit(0)
'''

if __name__ == "__main__":
    main()
