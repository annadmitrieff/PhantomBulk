#!/usr/bin/env python3
#ppd-physics.py
# Python script responsible for generating physically
# realistic parameters for protoplanetary disc simulations.

import argparse
import logging
import numpy as np
import pandas as pd
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
import sys

# ==================
# ++ DATA CLASSES ++
# ==================

@dataclass
class PPDParameters:
    mass: float
    radius: float
    inclination: float 
    accr_radius: float
    j2_moment: float
    k_B: float = 1.380649e-16          # Boltzmann constant (erg/K)
    mu: float = 2.34                   # Mean molecular weight for molecular H2
    m_H: float = 1.6735575e-24         # Hydrogen mass (g)
    gamma: float = field(default=1.4)  # Can vary between 1.3-1.67 depending on conditions
    
    def __post_init__(self):
        # Minimum temperature based on CMB + typical ISM heating
        # Ref: Dullemond & Dominik 2004, A&A, 417, 159
        self.T_min = 10.0  # K
        self.u_min = (self.k_B * self.T_min) / ((self.gamma - 1) * self.mu * self.m_H)

class PhysicalPPDGenerator:
    def load_survey_distributions(self, config: dict):
        # Updated parameter ranges based on observations
        self.parameter_ranges = {
            'm1': {
                # Stellar mass ranges from brown dwarf to intermediate mass
                # Ref: Pascucci et al. 2016, ApJ, 831, 125
                'core': (0.08, 4.0),
                'tail': (0.05, 7.0)
            },
            'accr1': {
                # Stellar radius in solar units, varies with mass
                # Ref: Baraffe et al. 2015, A&A, 577, A42
                'core': (0.5, 3.0),
                'tail': (0.3, 5.0)
            },
            'disc_m_fraction': {   
                # Disc-to-star mass ratio
                # Ref: Andrews et al. 2013, ApJ, 771, 129
                'core': (0.001, 0.2),
                'tail': (0.0005, 0.3)
            },
            'R_out': {
                # Outer radius in AU
                # Ref: Andrews et al. 2018, ApJ, 869, L41
                'core': (100, 300),
                'tail': (50, 500)
            },
            'H_R': {
                # Aspect ratio at reference radius
                # Ref: Dullemond & Dominik 2004, A&A, 421, 1075
                'core': (0.02, 0.15),
                'tail': (0.01, 0.25)
            },
            'dust_to_gas': {
                # Dust-to-gas mass ratio
                # Ref: Williams & Best 2014, ApJ, 788, 59
                'core': (0.001, 0.1),
                'tail': (0.0001, 0.2)
            },
            'grainsize': {
                # Grain sizes in cm
                # Ref: Testi et al. 2014, Protostars and Planets VI
                'core': (1e-5, 1),
                'tail': (1e-6, 10)
            },
            'graindens': {
                # Grain density in g/cm**3
                # Ref: Woitke et al. 2016, A&A, 586, A103
                'core': (1.0, 5.0),
                'tail': (0.5, 8.0)
            },
            'beta_cool': {
                # Cooling parameter
                # Ref: Baehr et al. 2017, ApJ, 849, 111
                'core': (0.5, 50),      
                'tail': (0.1, 100)     
            },
            'J2_body1': {
                # Stellar quadrupole moment
                # Ref: Ward-Duong et al. 2021, AJ, 161, 70
                'core': (0.0, 0.02),
                'tail': (0.0, 0.05)
            }
        }

    def compute_temperature_structure(self, stellar_mass: float) -> tuple:
        # Compute disc temperature structure.
        # Temperature scaling with stellar mass and radius
        # Ref: Andrews & Williams 2005, ApJ, 631, 1134
        L_star = stellar_mass**3.5 # Approximate luminosity scaling
        
        # Temperature at 1 AU with realistic scatter
        log_T0 = np.random.normal(
            np.log(280) + 0.25 * np.log(L_star),
            0.3
        )
        T0 = np.exp(log_T0)
        
        # Temperature power law index
        # Ref: D'Alessio et al. 2001, ApJ, 553, 321
        q = np.random.normal(-0.5, 0.1)
        q = np.clip(q, -0.75, -0.25) # Physical limits from radiative equilibrium
        
        return T0, q

    def generate_planet_system(self, stellar_mass: float, disc_mass: float,
                             R_in: float, R_out: float) -> List[PlanetParameters]:
        # Generate a physically consistent planetary system.
        # Planet formation efficiency
        # Ref: Mulders et al. 2018, ApJ, 869, L41
        efficiency = np.random.lognormal(mean=np.log(0.1), sigma=1.0)
        max_total_planet_mass = disc_mass * efficiency
        
        # Number of planets based on stellar mass
        # Ref: Zhu & Wu 2018, AJ, 156, 92
        lambda_poisson = 2.0 * (stellar_mass/1.0)**0.5
        n_planets = np.random.poisson(lambda_poisson)
        n_planets = min(n_planets, 8) # Cap maximum number
        
        if n_planets == 0:
            return []
            
        # Generate planet parameters
        planets = []
        remaining_mass = max_total_planet_mass
        
        # Minimum separation in mutual Hill radii
        # Ref: Pu & Wu 2015, ApJ, 807, 44
        min_separation = 8.0
        
        available_radii = np.logspace(np.log10(R_in*1.5), np.log10(R_out*0.8), 1000)
        planet_radii = []
        
        # Place planets with mutual Hill radius consideration
        for _ in range(n_planets):
            if len(planet_radii) == 0:
                # Place first planet randomly
                radius = np.random.choice(available_radii)
                planet_radii.append(radius)
            else:
                # Find valid positions respecting Hill stability
                valid_positions = []
                for r in available_radii:
                    valid = True
                    for existing_r in planet_radii:
                        if abs(np.log10(r/existing_r)) < min_separation/3:
                            valid = False
                            break
                    if valid:
                        valid_positions.append(r)
                
                if not valid_positions:
                    break
                    
                radius = np.random.choice(valid_positions)
                planet_radii.append(radius)
        
        planet_radii.sort()
        
        # Generate planet masses following observed distributions
        # Ref: Cumming et al. 2008, PASP, 120, 531
        for radius in planet_radii:
            if remaining_mass <= 0:
                break
                
            # Mass scales with orbital distance
            # Ref: Mordasini et al. 2012, A&A, 547, A111
            max_mass = min(
                remaining_mass, 
                10.0 * (radius/30)**(-1.5) # Jupiter masses
            )
            
            mass = np.random.power(0.6) * max_mass # Power law distribution
            remaining_mass -= mass
            
            # Inclination distribution
            # Ref: Xie et al. 2016, PNAS, 113, 11431
            incl = np.random.rayleigh(1.5)
            incl = min(incl, 40) # Cap maximum inclination
            
            # Accretion radius scaled to Hill radius
            # Ref: Lissauer et al. 2011, ApJS, 197, 8
            hill_radius = radius * (mass/(3*stellar_mass))**(1/3)
            accr_radius = np.random.uniform(0.1, 0.3) * hill_radius
            
            # J2 moment considering planet mass and rotation
            # Ref: Ward & Hamilton 2004, AJ, 128, 2501
            j2_moment = np.random.lognormal(mean=np.log(0.01), sigma=0.5)
            
            planet = PlanetParameters(
                mass=mass,
                radius=radius,
                inclination=incl,
                accr_radius=accr_radius,
                j2_moment=j2_moment
            )
            planets.append(planet)
            
        return planets

    def generate_single_ppd(self) -> PPDParameters:
        # Generate a single physically consistent PPD.
        max_attempts = 10 # Limit the number of attempts to prevent infinite loops
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            # Generate stellar mass
            stellar_mass = sample_parameter(
                self.parameter_ranges['m1']['core'],
                self.parameter_ranges['m1']['tail']
            )
            logging.debug(f"Generated stellar_mass: {stellar_mass}")

            # Temperature structure
            T0, q = self.compute_temperature_structure(stellar_mass)
            logging.debug(f"Computed temperature structure: T0={T0}, q={q}")

            # Disc structure
            disc_mass, R_out, R_in, Sigma0, pindex = self.compute_disc_structure(stellar_mass, T0, q)
            logging.debug(f"Computed disc structure: disc_mass={disc_mass}, R_out={R_out}, R_in={R_in}, Sigma0={Sigma0}, pindex={pindex}")

            # Compute aspect ratio based on temperature structure and stellar mass
            H_R = self.compute_aspect_ratio(T0, stellar_mass, R_ref=1.0)
            logging.debug(f"Computed aspect ratio: H_R={H_R}")

            # Sample dust properties
            dust_to_gas = sample_parameter(
                self.parameter_ranges['dust_to_gas']['core'],
                self.parameter_ranges['dust_to_gas']['tail']
            )
            grain_size = sample_parameter(
                self.parameter_ranges['grainsize']['core'],
                self.parameter_ranges['grainsize']['tail']
            )
            graindens = sample_parameter(
                self.parameter_ranges['graindens']['core'],
                self.parameter_ranges['graindens']['tail']
            )
            logging.debug(f"Sampled dust properties: dust_to_gas={dust_to_gas}, grain_size={grain_size}, graindens={graindens}")

            # Cooling parameter
            beta_cool = sample_parameter(
                self.parameter_ranges['beta_cool']['core'],
                self.parameter_ranges['beta_cool']['tail']
            )
            logging.debug(f"Sampled beta_cool: {beta_cool}")

            # Sample J2_body1
            J2_body1 = sample_parameter(
                self.parameter_ranges['J2_body1']['core'],
                self.parameter_ranges['J2_body1']['tail']
            )
            logging.debug(f"Sampled J2_body1: {J2_body1}")

            # Generate planetary system
            planets = self.generate_planet_system(
                stellar_mass, disc_mass, R_in, R_out
            )
            logging.debug(f"Generated {len(planets)} planets")

            accr1 = 1  # Fixed accretion radius

            logging.debug(f"Computed accr1 (stellar accretion radius): {accr1}")

            # Create parameter object
            params = PPDParameters(
                m1=stellar_mass,
                accr1=accr1,
                J2_body1=J2_body1,
                disc_m=disc_mass,
                R_in=R_in,
                R_out=R_out,
                H_R=H_R,
                pindex=pindex,
                qindex=q,
                dust_to_gas=dust_to_gas,
                grainsize=grain_size,
                graindens=graindens,
                beta_cool=beta_cool,
                T0=T0,
                planets=planets
            )
            logging.debug(f"Created PPDParameters: {params}")

            # Validate parameters
            if self.validate(params):
                logging.debug("Parameters validated successfully.")
                return params
            else:
                logging.warning("Generated parameters failed validation. Regenerating.")

        raise ValueError("Failed to generate valid PPD parameters after multiple attempts.")

    def validate(self, params: PPDParameters) -> bool:
        """Validate generated parameters. Implement necessary checks."""
        # General Validation to Address Disc Structure:
        if params.R_in >= params.R_out:
            return False
        if not (0 <= params.J2_body1 <= 0.1):
            return False
        if params.disc_m <= 0:
            return False

        # Additional Validation
        if params.beta_cool <= 1:
            return False
        if params.H_R < self.H_R_min or params.H_R > self.H_R_max:
            return False
        if params.grainsize <= 0:
            return False
        if params.graindens <= 0:
            return False
        if params.dust_to_gas <= 0:
            return False
        if params.m1 <= 0:
            return False
        if params.accr1 <= 0 or params.accr1 > params.R_in:
            return False  # Ensure accr1 is less than or equal to R_in

        # Validate temperature within a reasonable range
        T0_min = self.T0_min
        T0_max = 2000  # Kelvin 
        if not (T0_min <= params.T0 <= T0_max):
            return False

        # Validate planet parameters
        for planet in params.planets:
            if planet.accr_radius <= 0 or planet.accr_radius > 0.1:
                return False
            if planet.mass <= 0 or planet.mass > 10.0:
                return False  # Adjusted maximum planet mass
            if planet.radius <= params.R_in or planet.radius >= params.R_out:
                return False  # Ensure planets are within the disc
            if planet.radius - params.R_in < 0.1 * (params.R_out - params.R_in):
                return False  # Avoid placing planets too close to R_in

        return True

    def generate_parameter_set(self, n_discs: int) -> List[PPDParameters]:
        """Generate parameters for multiple discs."""
        return [self.generate_single_ppd() for _ in range(n_discs)]

# ==================
# ++ FILE MANAGER ++
# ==================

class PHANTOMFileManager:
    """Manage PHANTOM input file generation and modification."""

    def __init__(self, setup_template_path: str, PHANTOM_DIR: Optional[str] = None):
        self.setup_template = self.read_file(setup_template_path)
        self.PHANTOM_DIR = self.find_phantom_executables(PHANTOM_DIR)

    @staticmethod
    def read_file(filename: str) -> str:
        """Read the content of a file."""
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Template file '{filename}' not found.")
        with open(filename, 'r') as f:
            return f.read()

    def find_phantom_executables(self, custom_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Search for 'phantom' and 'phantomsetup' executables.
        If 'custom_dir' is provided, search there first.
        """
        executables = ["phantom", "phantomsetup"]
        executable_paths = []

        # If a custom directory is provided, check it first
        if custom_dir:
            custom_dir_path = Path(custom_dir).expanduser()
            for exe in executables:
                exe_path = custom_dir_path / exe
                if exe_path.is_file() and os.access(exe_path, os.X_OK):
                    executable_paths.append(exe_path)
                else:
                    logging.warning(f"Executable '{exe}' not found or not executable in '{custom_dir_path}'.")
        
        # Search in system PATH
        for exe in executables:
            if len(executable_paths) >= len(executables):
                break  # Already found all executables
            if exe not in [path.name for path in executable_paths]:
                path = shutil.which(exe)
                if path:
                    executable_paths.append(Path(path))
                else:
                    logging.error(f"Executable '{exe}' not found in system PATH.")
        
        if len(executable_paths) != len(executables):
            raise FileNotFoundError("One or more PHANTOM executables not found.")

        return tuple(executable_paths)  # Returns (phantom_path, phantomsetup_path)

    def create_setup_file(self, params: PPDParameters, output_dir: Path, sim_id: int):
        """Generate the `.setup` file with all placeholders replaced by params."""
        # Start with the setup template
        setup_content = self.setup_template

        # Number of planets
        num_planets = len(params.planets)

        # Generate planet configurations
        planet_configurations = self.generate_planet_configurations(params.planets)

        # Create dictionary for all parameter placeholders
        param_dict = {
            "m1": params.m1,
            "accr1": params.accr1,
            "J2_body1": params.J2_body1,
            "R_in": params.R_in,
            "R_out": params.R_out,
            "disc_m": params.disc_m,
            "pindex": params.pindex,
            "qindex": params.qindex,
            "H_R": params.H_R,
            "dust_to_gas": params.dust_to_gas,
            "grainsize": params.grainsize,
            "graindens": params.graindens,
            "beta_cool": params.beta_cool,
            "T0": params.T0,
            "NUM_PLANETS": num_planets,
            "PLANET_CONFIGURATIONS": planet_configurations
        }

        # Replace each placeholder with actual parameter values
        for key, value in param_dict.items():
            placeholder = f"{{{{{key}}}}}"  # Matches {{PARAM_NAME}} in templates
            setup_content = setup_content.replace(placeholder, str(value))

        # Write the fully populated .setup file
        setup_file = output_dir / 'dustysgdisc.setup'
        with open(setup_file, 'w') as f:
            f.write(setup_content)

    def generate_planet_configurations(self, planets: List[PlanetParameters]) -> str:
        """Generate the planet configuration strings for the .setup file."""
        planet_configs = ""
        for i, planet in enumerate(planets, 1):
            planet_configs += f"""
# planet:{i}
mplanet{i} =       {planet.mass:.3f}    ! planet mass (in Jupiter masses)
rplanet{i} =       {planet.radius:.3f}      ! orbital radius (in AU)
inclplanet{i} =    {planet.inclination:.3f}  ! orbital inclination (degrees)
accrplanet{i} =    {planet.accr_radius:.3f}  ! accretion radius (in Hill radii)
J2_body{i} =       {planet.j2_moment:.3f}    ! J2 moment (oblateness)
"""
        return planet_configs

    def modify_in_file(self, params: PPDParameters, output_dir: Path):
        """Modify the `.in` file with additional parameters after `phantomsetup` generates it."""
        in_file_path = output_dir / 'dustysgdisc.in'

        if not in_file_path.is_file():
            raise FileNotFoundError(f"'{in_file_path}' does not exist.")

        # Backup the original .in file
        backup_file_path = output_dir / 'dustysgdisc.in.bak'
        shutil.copy(in_file_path, backup_file_path)
        logging.info(f"Backup of the original .in file created at {backup_file_path}.")

        # Read current content
        with open(in_file_path, 'r') as f:
            in_content = f.read()

        # Define patterns and replacements
        replacements = {
            r'^\s*beta_cool\s*=\s*\d+\.\d+\s*!.*$': f"beta_cool = {params.beta_cool:.3f}    ! beta factor in Gammie (2001) cooling",
            r'^\s*T0\s*=\s*\d+\.\d+\s*!.*$': f"T0 = {params.T0:.3f}    ! Temperature at 1 AU"
        }

        # Perform replacements using regex
        for pattern, replacement in replacements.items():
            in_content, count = re.subn(pattern, replacement, in_content, flags=re.MULTILINE)
            if count > 0:
                logging.info(f"Replaced '{pattern}' with '{replacement}' in {in_file_path}.")
            else:
                logging.warning(f"Pattern '{pattern}' not found in {in_file_path}. Line not replaced.")

        # Write the modified .in file
        with open(in_file_path, 'w') as f:
            f.write(in_content)

# ==============================
# ++ GENERATE PHANTOM INPUT ++
# ==============================

def generate_phantom_input(params: PPDParameters, output_dir: Path, sim_id: int, file_manager: PHANTOMFileManager, config: dict) -> bool:
    """Generate PHANTOM setup file, run phantomsetup, modify .in, and create submission script."""

    # Step 1: Locate `phantom` and `phantomsetup` executables
    phantom_exe, phantomsetup_exe = file_manager.PHANTOM_DIR

    # Step 2: Generate the populated `.setup` file
    file_manager.create_setup_file(params, output_dir, sim_id)

    # Step 3: Ensure executables are executable
    phantom_exe.chmod(phantom_exe.stat().st_mode | 0o111)
    phantomsetup_exe.chmod(phantomsetup_exe.stat().st_mode | 0o111)

    # Step 4: Run `phantomsetup` to generate `dustysgdisc.in`
    subprocess.run([str(phantomsetup_exe), 'dustysgdisc'], cwd=output_dir) # Run twice to populate unspecified inputs with phantom calculations
    result_phantomsetup = subprocess.run([str(phantomsetup_exe), 'dustysgdisc'], cwd=output_dir)
    if result_phantomsetup.returncode != 0:
        logging.error(f"Error: 'phantomsetup' failed for simulation {sim_id}. Skipping.")
        return False

    # Step 5: Modify `.in` file based on additional parameters
    try:
        file_manager.modify_in_file(params, output_dir)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Skipping simulation {sim_id}.")
        return False

    # Step 5: Generate the submission script
    submission_script = f"""#!/bin/bash
#SBATCH --job-name=ppd_{sim_id:04d}                                # Job name
#SBATCH --partition={config.get('PARTITION'), 'batch'}             # Partition name
#SBATCH --ntasks={config.get('N_TASKS'), '1'}                      # 1 task (process)
#SBATCH --cpus-per-task={config.get('CPUS_PER_TASK'), '20'}        # CPU core count per task
#SBATCH --mem={config.get('MEM'), '10G'}                           # Memory per node
#SBATCH --time={config.get('TIME'), '6-23:59:59'}                  # Time limit (days-hours:minutes:seconds)
#SBATCH --output={output_dir}/ppd_{sim_id:04d}_%j.out              # Standard output log
#SBATCH --mail-user={config.get('USER_EMAIL', 'user@example.com')} # User email
#SBATCH --mail-type={config.get('MAIL_TYPE'), ""}                  # Mail events (BEGIN, END, FAIL, ALL)

# PHANTOM contingencies
export SYSTEM=gfortran
ulimit -s unlimited
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=28
export OMP_STACKSIZE=1024M

# Change to the simulation directory
cd {output_dir}

# Run PHANTOM with the input file
./phantom dustysgdisc.in
"""

    # Write the submission script
    submit_script_path = output_dir / f'run_{sim_id}.sh'
    with open(submit_script_path, 'w') as f:
        f.write(submission_script)

    # Make the submission script executable
    submit_script_path.chmod(submit_script_path.stat().st_mode | 0o111)
    logging.debug(f"Created submission script '{submit_script_path}'.")

    return True

# ==========
# ++ MAIN ++
# ==========

def main():
    # Main function to generate PPD simulations.
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Protoplanetary Disc Simulations")

    # Positional argument for number of simulations
    parser.add_argument('n_sims', type=int, help='Number of simulations to generate')

    # Optional arguments for flexibility
    parser.add_argument('-d', '--output_dir', type=str, default='$HOME/PhantomBulk/outputs/',
                        help='Output directory for simulations (default: PhantomBulk/outputs/)')
    parser.add_argument('-c', '--config_file', type=str, default='$HOME/PhantomBulk/setup/config.yaml',
                        help='Path to configuration file (YAML format, default: PhantomBulk/setup/config.yaml)')
    
    args = parser.parse_args()

    # Load configuration from file if provided
    config = {}
    if args.config_file:
        config_path = Path(args.config_file).expanduser()
        if config_path.is_file():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from '{config_path}'.")
        else:
            logging.error(f"Configuration file '{config_path}' not found.")
            sys.exit(1)
    else:
        logging.info("No configuration file provided. Using default settings.")

    # Assign arguments to variables
    n_sims = args.n_sims
    output_dir = Path(args.output_dir).expanduser()

    logging.info(f"Number of simulations to generate: {n_sims}")
    logging.info(f"Output directory: {output_dir}")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generators and managers
    generator = PhysicalPPDGenerator(config)

    # Initialize PHANTOMFileManager with the setup template path
    setup_template_path = Path(config.get('SETUP_TEMPLATE', 'dustysgdisc.setup')).expanduser()
    if not setup_template_path.is_file():
        logging.error(f"Setup template '{setup_template_path}' not found.")
        sys.exit(1)

    file_manager = PHANTOMFileManager(str(setup_template_path))

    # Create a submit_all.sh script
    submit_all_path = output_dir / 'submit_all.sh'
    with open(submit_all_path, 'w') as f:
        f.write('#!/bin/bash\n')

    # Make the submit_all.sh executable
    submit_all_path.chmod(submit_all_path.stat().st_mode | 0o111)
    logging.info(f"Created submission script '{submit_all_path}'.")

    # Generate parameters and setup/input files
    param_records = []
    for i in range(n_sims):
        # Generate physically consistent parameters
        try:
            params = generator.generate_single_ppd()
        except ValueError as e:
            logging.warning(f"Simulation {i}: {e}. Skipping.")
            continue

        # Set up simulation directory
        sim_dir = output_dir / f'sim_{i:04d}'
        sim_dir.mkdir(exist_ok=True)

        # Generate .setup file, copy executables, run phantomsetup, modify .in, and create submission script
        success = generate_phantom_input(params, sim_dir, i, file_manager, config)

        if not success:
            logging.warning(f"Simulation {i}: Failed to generate input files. Skipping.")
            continue

        # Add job to submit_all.sh
        with open(submit_all_path, 'a') as f:
            f.write(f'sbatch {sim_dir}/run_{i}.sh\n')

        # Record parameters in a dictionary
        param_dict = asdict(params)
        param_dict['simulation_id'] = i
        # Convert planets to list of dictionaries
        param_dict['n_planets'] = len(params.planets)
        param_dict['planets'] = str([asdict(planet) for planet in params.planets]) if params.planets else ''
        param_records.append(param_dict)

        # Optional: Print progress
        if (i+1) % 100 == 0 or (i+1) == n_sims:
            logging.info(f"Generated {i+1}/{n_sims} simulations")

    # Save parameters to CSV
    df = pd.DataFrame(param_records)
    param_db_path = output_dir / 'parameter_database.csv'
    df.to_csv(param_db_path, index=False)
    logging.info(f"Saved simulation parameters to '{param_db_path}'.")

    # Summary output
    logging.info(f"\nGenerated {len(param_records)} disc configurations")
    logging.info(f"Files saved in: {output_dir}")

    # ===================================
    # ++ INTERACTIVE SUBMISSION PROMPT ++
    # ===================================
    
    # Define submission command based on scheduler
    scheduler = config.get('scheduler', 'SLURM').upper()
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

    # Prompt the user to decide whether to submit all jobs
    print("")
    print("==============================================")
    print(f"You have generated {n_sims} simulations in '{output_dir}'.")
    print("Would you like to submit all jobs now?")
    print("It's recommended to verify the '.setup' and '.in' files before submission.")
    print("==============================================")
    while True:
        yn = input("Do you want to execute 'submit_all.sh' and submit all jobs? [y/n]: ").strip().lower()
        if yn in ['y', 'yes']:
            # User chose to submit all jobs
            logging.info(f"Submitting all jobs in '{output_dir}' using scheduler '{scheduler}'...")
            with open(submit_all_path, 'r') as f:
                for line in f:
                    job_script = line.strip()
                    if Path(job_script).is_file():
                        result = subprocess.run(job_scheduler_map[scheduler].split() + [job_script])
                        if result.returncode != 0:
                            logging.error(f"Job submission failed for script '{job_script}'.")
                        else:
                            logging.info(f"Submitted job: {job_script}")
                    else:
                        logging.warning(f"Job script '{job_script}' does not exist. Skipping.")
            logging.info("All simulations have been submitted successfully.")
            break
        elif yn in ['n', 'no', '']:
            # User chose not to submit jobs
            logging.info("Job submission skipped. You can verify the '.setup' and '.in' files in the output directory before submitting manually.")
            logging.info(f"To submit all jobs later, execute the 'submit_all.sh' script in '{output_dir}'.")
            break
        else:
            # Invalid input; prompt again
            print("Please answer yes (y) or no (n).")

    sys.exit(0)

if __name__ == "__main__":
    main()
