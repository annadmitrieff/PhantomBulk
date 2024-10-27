#!/bin/bash

# Description:
# This script generates protoplanetary disk simulations and submits them to Sapelo2.

# ====================
# PARAMETERS FOR JOB |
# ====================

# Default Number of simulations to generate
DEFAULT_N_SIMS=100

# Default Output Directory
DEFAULT_OUTPUT_DIR="/scratch/adm61595/adm61595/0_sink"

# Parse command-line arguments for flexibility
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--n_sims) N_SIMS="$2"; shift ;;
        -d|--output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set defaults if not provided
N_SIMS=${N_SIMS:-$DEFAULT_N_SIMS}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

# Path to the main Python script
PYTHON_SCRIPT="/home/adm61595/CHLab/1_HCA_PPDs/1_Code/3_RandomSims/ppd-physics.py"

# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Activates virtual environment
source /home/adm61595/Envs/PHANTOMEnv/bin/activate

# Run the main Python script to generate simulation configurations and job scripts
echo "Generating $N_SIMS simulations in directory '$OUTPUT_DIR'..."
python "$PYTHON_SCRIPT" "$N_SIMS" --output_dir "$OUTPUT_DIR"

# Check if the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: Python script execution failed!"
    exit 1
fi

# Path to the submit_all.sh script
SUBMIT_ALL_SCRIPT="${OUTPUT_DIR}/submit_all.sh"

# Ensure the submit_all.sh script was created
if [ ! -f "$SUBMIT_ALL_SCRIPT" ]; then
    echo "Error: Submission script '$SUBMIT_ALL_SCRIPT' not found!"
    exit 1
fi

# Make sure submit_all.sh is executable
chmod +x "$SUBMIT_ALL_SCRIPT"

# Submit all jobs using submit_all.sh
echo "Submitting all jobs in '$OUTPUT_DIR'..."
bash "$SUBMIT_ALL_SCRIPT"

# Check if jobs were submitted successfully
if [ $? -ne 0 ]; then
    echo "Error: Job submission failed!"
    exit 1
fi

echo "All simulations have been submitted successfully."
