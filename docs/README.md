

The goal of this tool is to functionally generate n number of simulations (dustysgdiscs) within a viable parameter space.
This is in order to realistically simulate the space of PPDs in space, without displaying bias toward any particular parameter, and yielding a normal distribution of sorts with respect to the most 'common' variety of PPDs based on characteristic physical parameters.

## =================
## ++ THE SCRIPTS ++
## =================

`master-script.sh` is the main control script

`ppd-physics.py` contains the PPD Generator class for generating parameters

## =================
## ++ SETUP STEPS ++
## =================

Create the job directory structure:

```
$mkdir phantom_simulations
$cd phantom_simulations
```

Copy all files into this directory.

Make the scripts executable:

`$chmod +x master-script.sh`

Run the master script...

Using default values (100 simulations, /scratch/0_sink/ directory)
`$`./master-script.sh``

Specifying number of simulations and output directory
`$`./master-script.sh --n_sims 500 --output_dir my_phantom_runs``

Using short flags
`$`./master-script.sh -n 500 -d my_phantom_runs``
