#!/bin/bash
# master-script.sh
# Master script responsible for:
    # config.yaml parsing
    # virtual environment setup
    # PHANTOM setup and input file generation
    # Optional job submission

# ========================
# ++ DEFAULT PARAMETERS ++
# ========================

# Default values
DEFAULT_N_SIMS=10
DEFAULT_OUTPUT_DIR="$HOME/PhantomBulk/outputs/"
DEFAULT_SCHEDULER="SLURM"
DEFAULT_SETUP_TEMPLATE="$HOME/PhantomBulk/templates/dustysgdisc.setup"
DEFAULT_PYTHON_SCRIPT="$HOME/PhantomBulk/src/main.py"
DEFAULT_VENV_DIR="$HOME/PhantomBulk/env/venv"
DEFAULT_REQUIREMENTS="$HOME/PhantomBulk/env/requirements.txt"
DEFAULT_CONFIG_FILE="$HOME/PhantomBulk/config/config.yaml"
CONFIG_TEMPLATE="$HOME/PhantomBulk/example/config.yaml.example"

# ========================
# ++ LOAD CONFIGURATION ++
# ========================

# Function to create config.yaml from template
create_config() {
    echo "Creating configuration file at '$DEFAULT_CONFIG_FILE'..."
    if [ ! -f "$CONFIG_TEMPLATE" ]; then
        echo "Error: Configuration template '$CONFIG_TEMPLATE' not found."
        exit 1
    fi
    cp "$CONFIG_TEMPLATE" "$DEFAULT_CONFIG_FILE"
    echo "Configuration file created. Please review and modify it as needed at '$DEFAULT_CONFIG_FILE'."
}

# Check if config.yaml exists
if [ ! -f "$DEFAULT_CONFIG_FILE" ]; then
    echo "Configuration file '$DEFAULT_CONFIG_FILE' not found."
    read -p "Would you like to create one now from the example? [y/n]: " yn
    case $yn in
        [Yy]* )
            create_config
            ;;
        [Nn]* )
            echo "Cannot proceed without a configuration file. Exiting."
            exit 1
            ;;
        * )
            echo "Please answer yes (y) or no (n). Exiting."
            exit 1
            ;;
    esac
fi

# Load configuration file
echo "Loading configuration from $DEFAULT_CONFIG_FILE..."
# Export variables from the config file
while IFS=: read -r key value; do
    # Ignore comments and empty lines
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    key=$(echo "$key" | tr -d ' ')
    value=$(echo "$value" | sed 's/^ *//;s/ *$//')
    export "$key"="$value"
done < "$DEFAULT_CONFIG_FILE"

# ==============================
# ++ PARSE COMMAND-LINE ARGS ++
# ==============================

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -n, --n_sims           Number of simulations to generate (default: $DEFAULT_N_SIMS)"
    echo "  -d, --output_dir       Output directory for simulations (default: $DEFAULT_OUTPUT_DIR)"
    echo "  -s, --scheduler        Job scheduler to use (SLURM, PBS, SGE) (default: $DEFAULT_SCHEDULER)"
    echo "  -t, --setup_template   Path to setup template file (default: $DEFAULT_SETUP_TEMPLATE)"
    echo "  -p, --python_script    Path to the main Python script (default: $DEFAULT_PYTHON_SCRIPT)"
    echo "  -v, --venv_dir         Path to the Python virtual environment directory (default: $DEFAULT_VENV_DIR)"
    echo "  -c, --config_file      Path to a custom configuration file"
    echo "  -h, --help             Display this help message"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--n_sims)
            N_SIMS="$2"
            shift 2
            ;;
        -d|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--scheduler)
            JOB_SCHEDULER="$2"
            shift 2
            ;;
        -t|--setup_template)
            SETUP_TEMPLATE="$2"
            shift 2
            ;;
        -p|--python_script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        -v|--venv_dir)
            VENV_DIR="$2"
            shift 2
            ;;
        -c|--config_file)
            CONFIG_FILE="$2"
            if [ -f "$CONFIG_FILE" ]; then
                echo "Loading configuration from $CONFIG_FILE..."
                while IFS=: read -r key value; do
                    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
                    key=$(echo "$key" | tr -d ' ')
                    value=$(echo "$value" | sed 's/^ *//;s/ *$//')
                    export "$key"="$value"
                done < "$CONFIG_FILE"
            else
                echo "Configuration file '$CONFIG_FILE' not found. Continuing with existing settings."
            fi
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# Set defaults if not provided via arguments or config
N_SIMS=${N_SIMS:-$DEFAULT_N_SIMS}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
JOB_SCHEDULER=${JOB_SCHEDULER:-$DEFAULT_SCHEDULER}
SETUP_TEMPLATE=${SETUP_TEMPLATE:-$DEFAULT_SETUP_TEMPLATE}
PYTHON_SCRIPT=${PYTHON_SCRIPT:-$DEFAULT_PYTHON_SCRIPT}
VENV_DIR=${VENV_DIR:-$DEFAULT_VENV_DIR}
REQUIREMENTS=${REQUIREMENTS:-$DEFAULT_REQUIREMENTS}

# ========================
# ++ VALIDATE INPUTS ++
# ========================

# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Ensure the setup template exists
if [ ! -f "$SETUP_TEMPLATE" ]; then
    echo "Error: Setup template '$SETUP_TEMPLATE' not found!"
    exit 1
fi

# Ensure the requirements.txt exists
if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: Requirements file '$REQUIREMENTS' not found!"
    exit 1
fi

# ===============================
# ++ VIRTUAL ENVIRONMENT SETUP ++
# ===============================

# Function to create virtual environment
create_virtualenv() {
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
}

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment directory '$VENV_DIR' does not exist."
    create_virtualenv
else
    echo "Virtual environment directory '$VENV_DIR' already exists."
fi

# Activate Virtual Environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi
echo "Virtual environment activated."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required Python packages from '$REQUIREMENTS'..."
pip install -r "$REQUIREMENTS"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install required Python packages."
    deactivate
    exit 1
fi
echo "Required Python packages installed successfully."

# =========================
# ++ GENERATE SIMULATIONS ++
# =========================

echo "Generating $N_SIMS simulations using setup template '$SETUP_TEMPLATE'..."
phantombulk --n_sims "$N_SIMS" --output_dir "$OUTPUT_DIR" --config_file "$DEFAULT_CONFIG_FILE"
PYTHON_EXIT_CODE=$?

# Deactivate virtual environment after running the script
deactivate

# Check if the Python script executed successfully
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Error: Python script execution failed!"
    exit 1
fi

# =======================
# ++ DEFINE SUBMISSION CMD ++
# =======================

# Define submission command based on scheduler
scheduler=$(echo "$JOB_SCHEDULER" | tr '[:lower:]' '[:upper:]')
case "$scheduler" in
    SLURM)
        SUBMIT_CMD="sbatch"
        ;;
    PBS)
        SUBMIT_CMD="qsub"
        ;;
    SGE)
        SUBMIT_CMD="qsub -cwd"
        ;;
    *)
        echo "Unsupported job scheduler: $JOB_SCHEDULER"
        exit 1
        ;;
esac

echo "Using job scheduler: $scheduler"

# ==============================
# ++ VERIFY SUBMISSION SCRIPT ++
# ==============================

# Path to the submit_all.sh script
SUBMIT_ALL_SCRIPT="${OUTPUT_DIR}/submit_all.sh"

# Ensure the submit_all.sh script was created
if [ ! -f "$SUBMIT_ALL_SCRIPT" ]; then
    echo "Error: Submission script '$SUBMIT_ALL_SCRIPT' not found!"
    exit 1
fi

# Make sure submit_all.sh is executable
chmod +x "$SUBMIT_ALL_SCRIPT"
echo "Submission script '$SUBMIT_ALL_SCRIPT' is ready."

# ===================================
# ++ INTERACTIVE SUBMISSION PROMPT ++
# ===================================

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
            echo ""
            echo "Submitting all jobs in '$OUTPUT_DIR' using scheduler '$scheduler'..."
            while IFS= read -r job_script; do
                if [ -f "$job_script" ]; then
                    $SUBMIT_CMD "$job_script"
                    if [ $? -ne 0 ]; then
                        echo "Error: Job submission failed for script '$job_script'."
                    else
                        echo "Submitted job: $job_script"
                    fi
                else
                    echo "Warning: Job script '$job_script' does not exist. Skipping."
                fi
            done < "$SUBMIT_ALL_SCRIPT"
            echo "All simulations have been submitted successfully."
            break
            ;;
        [Nn]* | "" )
            echo ""
            echo "Job submission skipped. You can verify the '.setup' and '.in' files in '$OUTPUT_DIR' before submitting manually."
            echo "To submit all jobs later, execute the 'submit_all.sh' script in '$OUTPUT_DIR'."
            break
            ;;
        * )
            echo "Please answer yes (y) or no (n)."
            ;;
    esac
done

exit 0
