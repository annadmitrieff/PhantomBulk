This toolkit provides a set of scripts and utilities to generate, run, and post-process simulations of protoplanetary disks using the PHANTOM and MCFOST software packages on high-performance computing (HPC) systems managed by SLURM.

This documentation will guide you through the setup, usage, and customization of the toolkit to help you run your own simulations.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Generating Simulations](#1-generating-simulations)
  - [2. Submitting Simulations](#2-submitting-simulations)
  - [3. Post-processing Results](#3-post-processing-results)
- [Configuration](#configuration)
- [Scripts Overview](#scripts-overview)
  - [`master-script.sh`](#master-scriptsh)
  - [`ppd-physics.py`](#ppd-physicspy)
  - [`post-process.sh`](#post-processsh)
- [Customization](#customization)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Automated Simulation Generation**: Generate physically realistic parameters for protoplanetary disk simulations.
- **Batch Job Submission**: Automatically create and submit batch jobs to a SLURM-managed HPC cluster.
- **Post-processing Utilities**: Process simulation outputs and prepare data for analysis.
- **Customizable Parameters**: Easily adjust simulation parameters and configurations.
- **Modular Design**: Scripts are organized for clarity and ease of use.

---

## Prerequisites

Before using the toolkit, ensure you have the following:

- Access to an HPC system managed by SLURM.
- Installed versions of **PHANTOM** and **MCFOST** software packages.
- **Python 3.6** or higher.
- Necessary Python packages:
  - `numpy`
  - `pandas`
  - `scipy`
  - `argparse`
  - `logging`

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/annadmitrieff/PhantomBulk.git
cd PhantomBulk
```

### 2. Set Up the Environment

It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
protoplanetary-disk-simulation-toolkit/
├── README.md
├── LICENSE
├── src/
│   ├── master-script.sh
│   ├── ppd-physics.py
│   └── post-process.sh
├── setup/
│   ├── dustysgdisc.setup
│   ├── phantom
│   ├── phantomsetup
│   └── ref4.1.para
├── utils/
│   └── SimPlots.ipynb
└── docs/
    └── documentation.md
    └── requirements.txt
```

- **`src/`**: Contains the main scripts for simulation generation, execution, and post-processing.
- **`setup/`**: Includes setup files and templates required for simulations.
- **`utils/`**: Utility scripts and notebooks for data analysis and visualization.
- **`docs/`**: Additional documentation and resources.

---

## Usage

### 1. Generating Simulations

The `ppd-physics.py` script generates simulation configurations and prepares them for execution.
This script and its processes are managed and executed using `master-script.sh`.

**Command Syntax:**

```bash
./master-script.sh -n [number of simulations] -d [output directory]
```

**Options:**

- `-n`, `--n_sims`: Specify the number of simulations to generate (default: 100).
- `-d`, `--output_dir`: Specify the output directory for simulations (default: `/scratch/0_sink`).

**Example:**

```bash
bash src/master-script.sh -n 100 -d /scratch/simulations
```

This command generates 100 simulations and stores them in `/scratch/simulations`.

After generating simulations, you can submit them to the SLURM scheduler in two ways.

**Interactive Submission:**

```
You have generated 100 simulations in '/scratch/simulations'.
Would you like to submit all jobs now?
It's recommended to verify the '.setup' and '.in' files before submission.
Do you want to execute 'submit_all.sh' and submit all jobs? [y/n]:
```

- **Yes (`y`)**: Submits all jobs immediately.
- **No (`n`)**: Allows you to review files before manual submission.

**Manual Submission:**

If you opt out of the interactive submission, navigate to your `output_dir` and execute the generated script:

```bash
bash submit_all.sh
```

### 3. Post-processing Results

After simulations are complete, use `post-process.sh` to process the results.

**Submitting Post-processing Job:**

```bash
sbatch scripts/post-process.sh
```

Ensure that the `TARGET_DIR` and `REF_FILE` variables in `post-process.sh` are correctly set.

---

## Configuration

### Adjusting Simulation Parameters

- **`ppd-physics.py`**: Modify default parameter ranges and distributions in the script if needed.
- **Command-line Arguments**: Use command-line options to override default settings when running scripts.

### SLURM Job Parameters

- **Memory and CPU Allocation**: Adjust `#SBATCH` directives in the shell scripts to suit your HPC environment.
- **Email Notifications**: Update the `#SBATCH --mail-user` directive with your email address to receive job notifications.

### File Paths

- Ensure all file paths in the scripts are correct and accessible in your environment. Replace placeholder paths like `/home/yourusername/` with your actual directories.

---

## Scripts Overview

### `master-script.sh`

- **Purpose**: Automates the generation of simulations and optionally submits them to SLURM.
- **Usage**:

  ```bash
  bash scripts/master-script.sh -n [number_of_simulations] -d [output_directory]
  ```

- **Features**:
  - Activates a virtual environment.
  - Checks for required Python packages and installs them if missing.
  - Runs `ppd-physics.py` to generate simulations.
  - Provides an interactive prompt to submit all jobs.

### `ppd-physics.py`

- **Purpose**: Generates physically realistic parameters for protoplanetary disk simulations.

- **Features**:
  - Generates simulation configurations based on empirical distributions.
  - Creates necessary input files for PHANTOM.
  - Records parameters in a CSV database.

### `post-process.sh`

- **Purpose**: Processes simulation outputs and prepares data for analysis.
- **Usage**:

  ```bash
  sbatch scripts/post-process.sh
  ```

- **Features**:
  - Collects specified dump files from simulations.
  - Applies MCFOST processing commands.
  - Organizes results into a designated directory.

---

## Customization

### Adjusting Parameter Distributions

Modify the `PhysicalPPDGenerator` class in `ppd-physics.py` to change the sampling ranges and distributions for simulation parameters.

### Changing Simulation Templates

- **Setup Files**: Edit `setup/dustysgdisc.setup` to modify the simulation setup template.
- **Reference Files**: Update `setup/ref4.1.para` if you need different reference parameters for MCFOST.

### Modifying SLURM Directives

Edit the `#SBATCH` directives in the shell scripts to adjust:

- Job name
- Partition/queue
- Number of tasks and CPUs
- Memory allocation
- Time limits
- Output logs
- Email notifications

---

## Examples

### Generating and Submitting 50 Simulations

```bash
bash scripts/master-script.sh -n 50 -d /scratch/my_simulations
```

### Post-processing After Simulations Completion

```bash
sbatch scripts/post-process.sh
```

Ensure that `post-process.sh` has the correct paths set for your simulations.

---

## Troubleshooting

### Common Issues

- **Missing Executables**: Ensure that `phantom` and `phantomsetup` are accessible and have execution permissions.
- **File Not Found Errors**: Verify that all file paths in the scripts point to existing files and directories.
- **Permission Denied**: Check that you have the necessary permissions for the directories and files involved.

### Tips

- **Logging**: Review the output logs generated by SLURM jobs for detailed error messages.
- **Environment Variables**: Ensure that any required environment modules or variables are loaded in the scripts.
- **Resource Allocation**: Adjust memory and CPU allocations in SLURM directives if jobs fail due to resource constraints.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- **PHANTOM**: A Smoothed Particle Hydrodynamics (SPH) code for astrophysics.
- **MCFOST**: A Monte Carlo radiative transfer code.