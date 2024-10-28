#!/bin/bash

# Description:
# This script generates protoplanetary disk simulations and submits them to Sapelo2.

# ========================
# ++ PARAMETERS FOR JOB ++
# ========================

# Default Number of simulations to generate
DEFAULT_N_SIMS=100

# Default Output Directory
DEFAULT_OUTPUT_DIR="/scratch/"

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
PYTHON_SCRIPT="/home/adm61595/CHLab/PhantomBulk/ppd-physics.py"

# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Activating Virtual Environment
source /home/adm61595/Envs/PHANTOMEnv/bin/activate

# Ensure necessary Python packages are installed
REQUIRED_PKG=("numpy" "pandas" "plotly" "scipy")
for pkg in "${REQUIRED_PKG[@]}"; do
    if ! python -c "import $pkg" &> /dev/null; then
        echo "Package '$pkg' not found. Installing..."
        pip install "$pkg"
    fi
done

# Run the main Python script to generate simulation configurations and job scripts
echo "Generating $N_SIMS simulations in directory '$OUTPUT_DIR'..."
python3 "$PYTHON_SCRIPT" "$N_SIMS" --output_dir "$OUTPUT_DIR"

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

# ===================================
# ++ INTERACTIVE SUBMISSION PROMPT ++
# ===================================

# Prompt the user to decide whether to submit all jobs
echo ""
echo "=============================================="
echo "You have generated $N_SIMS simulations in '$OUTPUT_DIR'."
echo "Would you like to submit all jobs now?"
echo "It's recommended to verify the '.setup' and '.in' files before submission."
echo "=============================================="
while true; do
    read -p "Do you want to execute 'submit_all.sh' and submit all jobs? [y/n]: " yn
    case $yn in
        [Yy]* )
            # User chose to submit all jobs
            echo ""
            echo "Submitting all jobs in '$OUTPUT_DIR'..."
            bash "$SUBMIT_ALL_SCRIPT"
            
            # Check if jobs were submitted successfully
            if [ $? -ne 0 ]; then
                echo "Error: Job submission failed!"
                exit 1
            fi
            
            echo "All simulations have been submitted successfully."
            break
            ;;
        [Nn]* | "" )
            # User chose not to submit jobs
            echo ""
            echo "Job submission skipped. You can verify the '.setup' and '.in' files in '$OUTPUT_DIR' before submitting manually."
            echo "To submit all jobs later, execute the 'submit_all.sh' script in the output directory."
            break
            ;;
        * )
            # Invalid input; prompt again
            echo "Please answer yes (y) or no (n)."
            ;;
    esac
done

exit 0