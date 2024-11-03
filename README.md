### PhantomBulk

This is set of tools for running and processing protoplanetary disk simulations in bulk using PHANTOM and MCFOST on SLURM-managed HPC systems.
Not for general use yet because I haven't generalized it, so half the code is specific to my filesystem...

#### Installation

Clone the respository (omit the $ when executing commands):

```bash
$ git clone https://github.com/annadmitrieff/PhantomBulk.git
```

#### The Scripts

master-script.sh is the main control script.

ppd-physics.py contains the PPD Generator class for generating parameters.

post-process.sh is a post-processing script for generating continuum images with MCFOST.

#### Creating Simulations

Create the job directory structure:
```bash
$ mkdir phantom_simulations
$ cd phantom_simulations
```
Copy all files into this directory.

Make the scripts executable:
```bash
$ chmod +x master-script.sh 
```
Run the master script...

Using default values (100 simulations, /scratch/ directory):
```bash
$ ./master-script.sh
```
Specifying number of simulations (500) and output directory (my_phantom_runs):
```bash
$ ./master-script.sh --n_sims 500 --output_dir my_phantom_runs
```
Using short flags:
```bash
$ ./master-script.sh -n 500 -d my_phantom_runs
```
Post-process your simulations (grabs 50th dump file):
```bash
sbatch post-process.sh
```
