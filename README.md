
# PhantomBulk

<img src="https://phantomsph.github.io/image/logo.png" alt="PhantomLogo" width="200" height="200"/>

PhantomBulk is a suite of tools designed for running and processing protoplanetary disk simulations in bulk using [PHANTOM](https://github.com/danieljprice/phantom) and [MCFOST](https://github.com/cpinte/mcfost) on SLURM-managed HPC systems.

I'm currently the sole writer/user so support is limited to what I need out of this program (there are likely many bugs as well).

## Features

- **SLURM Support:** Optimized for SLURM job schedulers. Future support for PBS and other schedulers is in progress.
- **Setup Configurations:** Currently only supports `dustysgdisc` pre-baked configuration. Future support for other pre-baked setups is in progress.
- **Automated Simulation Generation:** Easily generate and manage a large number of simulations.
- **Post-processing Utilities:** Tools for processing simulation outputs with MCFOST and other data analysis tools.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/annadmitrieff/PhantomBulk.git
cd PhantomBulk
```

### 2. Set Up a Python Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv env/venv
source env/venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `setup.py`:

```bash
pip install -e .
```

This will install `PhantomBulk` and its dependencies, and set up the `phantombulk` console script.

## Configuration

Before running simulations, configure the necessary parameters.

1. **Copy the Example Configuration File:**

   ```bash
   cp example/config.yaml.example config/config.yaml
   ```

2. **Edit `config/config.yaml`:**

   Open the configuration file in your preferred text editor and adjust the parameters as needed.

```yaml
# Paths
VENV_PATH: "$HOME/PhantomBulk/env"

PHANTOM_DIR: "$HOME/phantom"  # Change this if necessary (e.g., ~/Software/phantom/)
PYTHON_SCRIPT: "$HOME/PhantomBulk/src/PhantomBulk/main.py"

# Setup Templates
SETUP_TEMPLATE: "$HOME/PhantomBulk/templates/dustysgdisc.setup"
REFERENCE_FILE: "$HOME/PhantomBulk/templates/ref4.1.para"       

# PHANTOM Setup Type
PHANTOM_SETUP_TYPE: "default" # Options: default, setup1, setup2, etc.

# System & Job Information
JOB_SCHEDULER: "SLURM" 
CPUS_PER_TASK: "20"
PARTITION: "batch"
MAIL_TYPE: "FAIL"
N_TASKS: "1"
TIME: "6-23:59:59"
MEM: "10G"

# Your Information
USER_EMAIL: "annadmitrieff@uga.edu"

# Simulation Parameters
N_SIMS: 10
OUTPUT_DIR: "$HOME/PhantomBulk/outputs/"

# Logging
log_level: "INFO"

# Random Seed for Reproducibility
seed: 42

# Parameter Ranges
parameter_ranges:
  m1:
    core: [0.08, 4.0]
    tail: [0.05, 7.0]
  accr1: # placeholder range; this is *always* R_in - 0.01
    core: [0.02, 0.09]
    tail: [0.01, 0.1]
  disc_m_fraction:
    core: [0.001, 0.2]
    tail: [0.0005, 0.3]
  R_in:
    core: [0.03,0.1]
    tail: [0.02,0.2]
  R_out:
    core: [100, 300]
    tail: [50, 500]
  H_R:
    core: [0.02, 0.15]
    tail: [0.01, 0.25]
  dust_to_gas:
    core: [0.001, 0.1]
    tail: [0.0001, 0.2]
  grainsize:
    core: [1e-5, 1]
    tail: [1e-6, 10]
  graindens:
    core: [1.0, 5.0]
    tail: [0.5, 8.0]
  beta_cool:
    core: [0.5, 50]
    tail: [0.1, 100]
  J2_body1:
    core: [0.0, 0.02]
    tail: [0.0, 0.05]

```

## Usage

### Generating Simulations

Use the `phantombulk` console script to generate simulations.

#### Basic Usage

Generate simulations with default parameters (e.g., 10 simulations, output directory at `~/PhantomBulk/outputs/`):

```bash
phantombulk
```

#### Specifying Number of Simulations and Output Directory

Generate 100 simulations and specify a custom output directory (`my_phantom_runs`):

```bash
phantombulk --n_sims 100 --output_dir my_phantom_runs
```

#### Using Short Flags

```bash
phantombulk -n 500 -d my_phantom_runs
```

### Post-processing Simulations

After generating simulations, you can post-process them using the provided scripts. These are not routinely updated/are still largely customized to my HPC system (where MCFOST is installed weirdly)

#### Example: Post-process Your Simulations

Assuming you want to post-process and grab the 50th dump file, submit the post-processing job:

```bash
sbatch scripts/post-process.sh
```

## Additional Utilities

PhantomBulk includes utilities for data analysis and visualization using MCFOST and other tools. These utilities are available within the package and can be accessed as needed in `utils`.

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements. This probably needs many...

## License

[MIT License](LICENSE)

## Contact

For any questions or support, please contact me by [email](mailto:annadmitrieff@uga.edu)!
```

