#!/bin/bash

# ==============================
# PARAMETERS FOR POST-PROCESSING 
# ==============================

DEFAULT_TARGET_DIR="/scratch/"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--sim_path) SIM_PATH="$2"; shift ;;                          # Directory to find simulations
        -d|--output_dir) TARGET_DIR="$2"; shift ;;                      # Directory to output processed data
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set defaults (incl. if not provided)
TARGET_DIR=${TARGET_DIR:-$DEFAULT_TARGET_DIR}
REF_FILE="/home/adm61595/CHLab/PhantomBulk/src/ref4.1.para"

echo "Looking for simulations in ${SIM_PATH}..."
echo "Using MCFOST parameter file ${REF_FILE}..."
echo "Post-processed outputs in ${TARGET_DIR}..."

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# =========================
# FINDING DUMP, SETUP FILES
# =========================

echo "Grabbing files..."

# Step 1: Loop through each simulation directory to find every 50th dump file (keep dump files constant across sim sets)
for SIM_DIR in "${SIM_PATH}"/*; do # For each item in the simulation runs directory,

  if [ -d "${SIM_DIR}" ]; then  # If the item is a directory,
    
    if [ -f "${SIM_DIR}/dustysgdisc.setup" ]; then # And if the file 'dustysgdisc.setup' exists in the directory,

      if [ -f "${SIM_DIR}/dustysgdisc_00050" ]; then # And if the file 'dustysgdisc_00050' exists in the directory,

        SETUP_FILE="${SIM_DIR}/dustysgdisc.setup"

        DUMP_FILE="${SIM_DIR}/dustysgdisc_00050"

        SIM_NAME=$(basename "$SIM_DIR")

        DEST_DIR="$TARGET_DIR/$SIM_NAME"

        JOB_NAME="$DEST_DIR/$SIM_NAME-mcfost.sh"

        mkdir -p "$DEST_DIR"

        cp "$DUMP_FILE" "$DEST_DIR"
        cp "$SETUP_FILE" "$DEST_DIR"
        cp "$REF_FILE" "$DEST_DIR"

        echo "Working on $SIM_NAME. Creating $JOB_NAME in directory $DEST_DIR."

        echo "#!/bin/bash
#SBATCH --job-name=$SIM_NAME-mcfost			    # Job name (testBowtie2)
#SBATCH --partition=batch                   # Partition name (batch, highmem_p, or gpu_p)
#SBATCH --ntasks=1                          # 1 task (process) for below commands; may be allocated to different compute nodes (--nodes=1 negates that)
#SBATCH --cpus-per-task=8                   # CPU core count per task, by default 1 CPU core per task
#SBATCH --mem=5G                            # Memory per node (4GB); by default using M as unit
#SBATCH --time=6-23:59:59                   # Time limit hrs:min:sec or days-hours:minutes:seconds
#SBATCH --output=$DEST_DIR/%x_%j.out        # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=adm61595@uga.edu        # Where to send mail
#SBATCH --mail-type=FAIL                    # Mail events (BEGIN, END, FAIL, ALL)

cd $DEST_DIR

# EXECUTING MCFOST COMMANDS

$HOME/local/glibc-2.34/lib/ld-linux-x86-64.so.2 --library-path \"$HOME/local/glibc-2.34/lib:/lib64\" /home/adm61595/Software/mcfost $DEST_DIR/ref4.1.para -phantom $DEST_DIR/dustysgdisc_00050 -mol

$HOME/local/glibc-2.34/lib/ld-linux-x86-64.so.2 --library-path \"$HOME/local/glibc-2.34/lib:/lib64\" /home/adm61595/Software/mcfost $DEST_DIR/ref4.1.para -phantom $DEST_DIR/dustysgdisc_00050 -img 1300

echo "Processing completed. Moving files:"

DATA_DIR=\"$DEST_DIR/data_1300/\"

echo "Looking for .fits.gz files in $DATA_DIR..."

gunzip \${DATA_DIR}RT.fits.gz

echo "Unzipping .fits.gz file \${DATA_DIR}RT.fits.gz..."

mv data_1300/RT.fits $TARGET_DIR/$SIM_NAME.fits

echo "Moving \${DATA_DIR}RT.fits to $TARGET_DIR/$SIM_NAME.fits..."

echo "Files moved to $TARGET_DIR/$SIM_NAME.fits."
        " > $JOB_NAME

        sbatch $JOB_NAME
        echo "Post-processing job submitted."

      fi
    fi

    if [ ! -f "$DUMP_FILE" ]; then 
      echo "No dump file found in ${SIM_DIR}. Skipping."
    fi

  fi

done
