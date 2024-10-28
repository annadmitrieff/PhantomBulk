#!/bin/bash
#SBATCH --job-name=postprocessbulk			# Job name (testBowtie2)
#SBATCH --partition=batch                   # Partition name (batch, highmem_p, or gpu_p)
#SBATCH --ntasks=1                          # 1 task (process) for below commands; may be allocated to different compute nodes (--nodes=1 negates that)
#SBATCH --cpus-per-task=8                   # CPU core count per task, by default 1 CPU core per task
#SBATCH --mem=50G                           # Memory per node (4GB); by default using M as unit
#SBATCH --time=6-23:59:59                   # Time limit hrs:min:sec or days-hours:minutes:seconds
#SBATCH --output=%x_%j.out                  # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=adm61595@uga.edu        # Where to send mail
#SBATCH --mail-type=END,FAIL                # Mail events (BEGIN, END, FAIL, ALL)

source ~/.bashrc                            # Ensures ~/.bashrc file is sourced

# ============================
# PARAMETERS FOR POST-PROCESS |
# ============================

TARGET_DIR="/home/adm61595/runs/5_BulkSims/MCFOST"                        # Directory to store final results
REF_FILE="/home/adm61595/CHLab/PhantomBulk/setup/ref4.1.para"             # Reference parameter file

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Starting post-processing of simulation results..."

# Step 1: Loop through each simulation directory to find every 50th dump file
for SIM_DIR in "path_to_simulations"/simulations/*; do
  SIM_NAME=$(basename "$SIM_DIR")
  DUMP_FILE="${SIM_DIR}/dustysgdisc_00050"
  DEST_DIR="$TARGET_DIR/$SIM_NAME"

  # Create a subfolder for each simulation
  mkdir -p "$DEST_DIR"
  
  # Step 2: Copy the dump file and reference file into the designated directory
  cp "$DUMP_FILE" "$DEST_DIR/"
  cp "$REF_FILE" "$DEST_DIR/"

  # Step 3: Run MCFOST processing commands
  echo "Processing dump file for $SIM_NAME..."

  /home/adm61595/local/glibc-2.34/lib/ld-linux-x86-64.so.2 --library-path "/home/adm61595/local/glibc-2.34/lib:/lib64" \
    /home/adm61595/Software/mcfost "$DEST_DIR/ref4.1.para" -phantom "$DEST_DIR/dustysgdisc_00050" -mol

  /home/adm61595/local/glibc-2.34/lib/ld-linux-x86-64.so.2 --library-path "/home/adm61595/local/glibc-2.34/lib:/lib64" \
    /home/adm61595/Software/mcfost "$DEST_DIR/ref4.1.para" -phantom "$DEST_DIR/dustysgdisc_00050" -img 1300

  # Step 4: Unzip, rename, and organize the output file
  DATA_DIR="$DEST_DIR/data_1300"
  gunzip "$DATA_DIR/RT.fits.gz"
  mv "$DATA_DIR/RT.fits" "$TARGET_DIR/${SIM_NAME}.fits"
done

echo "Post-processing complete. Results are stored in $TARGET_DIR."
