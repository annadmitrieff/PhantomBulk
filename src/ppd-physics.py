#!/usr/bin/env python3

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

# ==================
# ++ DATA CLASSES ++
# ==================

@dataclass
class PlanetParameters:
    mass: float
    radius: float
    inclination: float
    accr_radius: float
    j2_moment: float

@dataclass
class PPDParameters:
    m1: float
    accr1: float
    J2_body1: float
    disc_m: float
    R_in: float
    R_out: float
    pindex: float
    qindex: float
    H_R: float
    dust_to_gas: float
    grainsize: float
    graindens: float
    beta_cool: float
    T0: float
    planets: List[PlanetParameters] = field(default_factory=list)
    # Internal energy per unit mass (u)
    u_min: float = field(init=False)

    def __post_init__(self):
        # Compute minimum internal energy per unit mass based on T0_min
        # Constants
        self.k_B = 1.380649e-16  # Boltzmann constant in erg/K
        self.mu = 2.34           # Mean molecular weight for molecular hydrogen
        self.m_H = 1.6735575e-24 # Mass of hydrogen atom in g
        self.gamma = 1.4         # Adiabatic index

        # Set minimum temperature in Kelvin
        self.T_min = 50  # Adjust as needed?

        # Compute minimum internal energy per unit mass (u_min)
        self.u_min = (self.k_B * self.T_min) / ((self.gamma - 1) * self.mu * self.m_H)

    def enforce_internal_energy_floor(self, u: float) -> float:
        # Ensure internal energy per unit mass is above the minimum.
        return max(u, self.u_min)

# =======================
# ++ SAMPLING FUNCTION ++
# =======================

def sample_parameter(core_range: Tuple[float, float],
                     tail_range: Tuple[float, float],
                     tail_probability: float = 0.1) -> float:
    """
    Sample a parameter with a probability to select from the tail range.

    Args:
        core_range (Tuple[float, float]): The (min, max) for the core range.
        tail_range (Tuple[float, float]): The (min, max) for the tail range.
        tail_probability (float): Probability of selecting from the tail range.

    Returns:
        float: Sampled parameter value.
    """
    if np.random.random() < tail_probability:
        return np.random.uniform(*tail_range)
    else:
        return np.random.uniform(*core_range)

# ==========================
# ++ PPD GENERATOR CLASS ++
# ==========================

class PhysicalPPDGenerator:
    # Generate PPD parameters with physical correlations
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        # Load empirical distributions from surveys
        self.load_survey_distributions()

    def load_survey_distributions(self):
        # Load empirical distributions from astronomical surveys
        # Stellar mass distribution (Kroupa IMF)
        self.imf_alpha = -2.3  # High-mass slope
        self.mass_break = 0.5  # Solar masses

        # Disk properties from surveys Andrews et al. 2010, Ansdell et al. 2016:
            # PROTOPLANETARY DISK STRUCTURES IN OPHIUCHUS. II. EXTENSION TO FAINTER SOURCES
            # ALMA SURVEY OF LUPUS PROTOPLANETARY DISKS. I. DUST AND GAS MASSES
        self.disk_params = {
            'mass_ratio_mean': np.log(0.01),  # Mean log disk-to-star mass ratio
            'mass_ratio_std': 0.8,            # Scatter in log mass ratio
            'size_mass_slope': 0.5,           # Power-law index for R_out vs M_disk
            'size_mass_scatter': 0.3          # Scatter in size-mass relation
        }

        # Temperature profile parameters
        self.temp_params = {
            'T0_mean': np.log(250),   # Mean log temperature at 1 AU
            'T0_std': 0.3,            # Scatter in log temperature
            'q_mean': -0.5,           # Mean temperature power-law index
            'q_std': 0.15             # Scatter in temperature index
        }
        self.T0_min = 10     # Minimum Temperature
        self.H_R_min = 0.03  # Minimum aspect ratio
        self.H_R_max = 0.25  # Maximum aspect ratio

        # Define core and tail ranges for each parameter:
        self.parameter_ranges = {
            'm1': {
                'core': (0.1, 3.0),
                'tail': (0.08, 5.0)
            },
            'accr1': {
                'core': (1, 1), # fixed accretion radius
                'tail': (1, 1)
            },
            'disc_m_fraction': {   
                'core': (0.001, 0.1),
                'tail': (0.0005, 0.2)
            },
            'R_out': {
                'core': (150, 250),
                'tail': (100, 300)
            },
            'H_R': {
                'core': (0.03, 0.2),
                'tail': (0.02, 0.25)
            },
            'dust_to_gas': { # ratio
                'core': (0.005, 0.05),
                'tail': (0.001, 0.1)
            },
            'grainsize': { # in cm
                'core': (0.0001, 0.1),
                'tail': (0.00001, 1.0)
            },
            'graindens': {
                'core': (1.5, 4.0),
                'tail': (1.0, 5.0)
            },
            'beta_cool': {
                'core': (1, 25),      
                'tail': (0.01, 75)     
            },
            'J2_body1': {
                'core': (0.0, 0.01),
                'tail': (0.01, 0.1)
            }
        }

    def generate_stellar_mass(self) -> float:
        # Generate stellar mass following the Initial Mass Function:
        while True:
            # Sample from broken power-law IMF
            if np.random.random() < 0.5:  # Low-mass segment
                mass = (np.random.random() * (self.mass_break**0.3))**(1/0.3)
            else:  # High-mass segment
                mass = self.mass_break * (np.random.random() *
                       (10**(-self.imf_alpha) - 1) + 1)**(1/(-self.imf_alpha))

            # Accept masses between 0.1 and 5 solar masses
            if 0.1 <= mass <= 5.0:
                return mass

    def compute_temperature_structure(self, stellar_mass: float) -> tuple:
        # Compute disk temperature structure:
        # Temperature at 1 AU (influenced by stellar mass)
        log_T0 = np.random.normal(
            self.temp_params['T0_mean'] + 0.7 * np.log(stellar_mass), # Mass-dependent scatter; logarithmic dependency scaling w/ mass
            self.temp_params['T0_std']
        )
        T0 = np.exp(log_T0)

        # Ensure T0 is above minimum
        T0 = max(T0, self.T0_min)

        q = 0.250

        return T0, q

    def compute_disk_structure(self, stellar_mass: float, T0: float, q: float) -> tuple:
        # Compute physically consistent disk structure
        # Sample disk mass with stellar mass correlation
        log_mass_ratio = np.random.normal(
            self.disk_params['mass_ratio_mean'] + 0.5 * np.log(stellar_mass),
            self.disk_params['mass_ratio_std']
        )
        disk_mass_fraction = np.exp(log_mass_ratio)
        disk_mass_fraction = np.clip(disk_mass_fraction, 0.0005, 0.2)
        disk_mass = stellar_mass * disk_mass_fraction

        R_out = sample_parameter(
                    self.parameter_ranges['R_out']['core'],
                    self.parameter_ranges['R_out']['tail']
        )

        R_in = 1

        p = 1

        # Disk mass fraction of stellar mass
        disk_mass_fraction = sample_parameter(
            self.parameter_ranges['disc_m_fraction']['core'],
            self.parameter_ranges['disc_m_fraction']['tail']
        )
        disk_mass_fraction = np.clip(disk_mass_fraction, 0.001, 0.1)
        disk_mass = stellar_mass * disk_mass_fraction

        # Calculate Sigma0 based on disk mass and radii
        if p != 2:
            factor = (R_out**(2 - p) - R_in**(2 - p)) / (2 - p)
        else:
            factor = np.log(R_out / R_in)

        Sigma0 = disk_mass / (2 * np.pi * factor)

        # Log computed values
        logging.debug(f"Computed disk mass: {disk_mass}, Sigma0: {Sigma0}")

        return disk_mass, R_out, R_in, Sigma0, p

    def compute_aspect_ratio(self, T0: float, stellar_mass: float, R_ref: float) -> float:
        # Compute aspect ratio H/R with physical dependencies
        # Constants in cgs units
        G = 6.67430e-8          # Gravitational Constant
        k_B = 1.380649e-16      # Boltzmann's Constant
        mu = 2.34               # Mean Molecular Weight
        m_H = 1.6735575e-24     # Hydrogen Mass
        AU = 1.496e13           # AU (cm)
        M_sun = 1.989e33        # Solar Mass (g)

        # Sound speed
        k_B = 1.380649e-16  # Boltzmann constant in erg/K
        mu = 2.34           # Mean molecular weight for molecular hydrogen
        m_H = 1.6735575e-24 # Mass of hydrogen atom in g
        c_s = np.sqrt(k_B * T0 / (mu * m_H))  # cm/s

        # Convert to cgs
        M_star = stellar_mass * M_sun
        R = R_ref * AU
        
        # Sound speed with temperature dependence?
        c_s = np.sqrt(k_B * T0 / (mu * m_H))
        
        # Keplerian velocity
        v_K = np.sqrt(G * M_star / R)
        
        # Base aspect ratio from hydrostatic equilibrium
        H_R = c_s / v_K
        
        # Add random fluctuations (turbulence, magnetic fields, etc.)
        fluctuation = np.random.normal(1.0, 0.1)
        H_R *= fluctuation
        
        # Ensure within physical limits
        H_R = np.clip(H_R, self.H_R_min, self.H_R_max)

        return H_R

    def generate_planet_system(self, stellar_mass: float, disk_mass: float,
                               R_in: float, R_out: float) -> List[PlanetParameters]:
        # Generate physically consistent planetary system
        # Determine number of planets with limits
        max_planets = min(6, int(disk_mass / 0.005))
        n_planets = np.random.randint(0, max_planets + 1)

        if n_planets == 0:
            return []

        # Set a margin to avoid planets lying exactly on R_in or R_out
        margin = 0.1 * (R_out - R_in)  # Increased margin to 10%

        # Generate planet locations within disk boundaries
        available_radii = np.linspace(R_in + margin, R_out - margin, 1000)
        planet_radii = np.sort(np.random.choice(available_radii, n_planets, replace=False))

        planets = []
        for radius in planet_radii:
            # Planet mass based on disk mass and location
            # Use isolation mass or Hill sphere considerations
            max_mass = min(
                10.0,  # Maximum planet mass (M_Jupiter)
                (disk_mass / 0.1) * (radius / 30)**(-1.5)  # Mass decreases with radius
            )
            mass = np.random.uniform(0.1, max_mass)

            # Inclination dependent on system mass
            max_incl = 5 * (stellar_mass / disk_mass)**0.2  # Reduced maximum inclination
            incl = np.random.rayleigh(max_incl / 3)
            incl = min(incl, 15)  # Enforce inclination limit

            # Sample accretion radius and J2 moment
            accr_radius = np.random.uniform(0.02, 0.05)  # Accretion radius in Hill radii
            j2_moment = np.random.uniform(0.0, 0.05)    # J2 moment

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
        # Generate a single physically consistent PPD
        max_attempts = 10  # Limit the number of attempts to prevent infinite loops...
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

            # Disk structure
            disk_mass, R_out, R_in, Sigma0, pindex = self.compute_disk_structure(stellar_mass, T0, q)
            logging.debug(f"Computed disk structure: disk_mass={disk_mass}, R_out={R_out}, R_in={R_in}, Sigma0={Sigma0}, pindex={pindex}")

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
                stellar_mass, disk_mass, R_in, R_out
            )
            logging.debug(f"Generated {len(planets)} planets")

            accr1 = 1

            logging.debug(f"Computed accr1 (stellar accretion radius): {accr1}")

            # Create parameter object
            params = PPDParameters(
                m1=stellar_mass,
                accr1=accr1,
                J2_body1=J2_body1,
                disc_m=disk_mass,
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
        # Checks and balances...
        if params.R_in >= params.R_out:
            return False
        if not (0 <= params.J2_body1 <= 0.1):
            return False
        if params.disc_m <= 0:
            return False
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
            return False  # Ensure accr1 is less than R_in

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
                return False  # Ensure planets are within the disk
            if planet.radius - params.R_in < 0.1 * (params.R_out - params.R_in):
                return False  # Avoid placing planets too close to R_in

        return True

    def generate_parameter_set(self, n_discs: int) -> List[PPDParameters]:
        # Generate parameters for multiple discs
        return [self.generate_single_ppd() for _ in range(n_discs)]

# ==================
# ++ FILE MANAGER ++
# ==================

class PHANTOMFileManager:
    # Manage PHANTOM input file generation and modification.

    def __init__(self, setup_template_path: str):
        self.setup_template = self.read_file(setup_template_path)

    @staticmethod
    def read_file(filename: str) -> str:
        # Read the content of a file.
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Template file '{filename}' not found.")
        with open(filename, 'r') as f:
            return f.read()

    def create_setup_file(self, params: PPDParameters, output_dir: Path, sim_id: int):
        # Generate the `.setup` file with all placeholders replaced by params.
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
        # Generate the planet configuration strings for the .setup file.
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

def generate_phantom_input(params: PPDParameters, output_dir: Path, sim_id: int, file_manager: PHANTOMFileManager) -> bool:
    # Generate PHANTOM setup file, run phantomsetup, modify .in, and create submission script.

    # Step 1: Copy `phantom` and `phantomsetup` executables
    src_dir = Path("/home/adm61595/CHLab/PhantomBulk/phantom/")
    executables = ["phantom", "phantomsetup"]
    for exe in executables:
        src_path = src_dir / exe
        dest_path = output_dir / exe
        if src_path.is_file():
            subprocess.run(["cp", str(src_path), str(dest_path)], check=True)
        else:
            logging.error(f"{exe} executable not found at {src_path}.")
            return False  # Skips this simulation if executables are missing

    # Step 2: Generate the populated `.setup` file
    file_manager.create_setup_file(params, output_dir, sim_id)

    # Step 3: Make the executables executable
    for exe in executables:
        dest_path = output_dir / exe
        dest_path.chmod(dest_path.stat().st_mode | 0o111)

    # Step 4: Run `phantomsetup` to generate `dustysgdisc.in`
    result_phantomsetup = subprocess.run(['./phantomsetup', 'dustysgdisc'], cwd=output_dir)
    if result_phantomsetup.returncode != 0:
        logging.error(f"Error: 'phantomsetup' failed for simulation {sim_id}. Skipping.")
        return False

    # Step 5: Modify `.in` file based on additional parameters
    try:
        # Run phantomsetup again if needed
        result_phantomsetup = subprocess.run(['./phantomsetup', 'dustysgdisc'], cwd=output_dir) # Runs it a second time to override missing values when dealing with nonzero J2 (auto-calculates)
        file_manager.modify_in_file(params, output_dir)
    except FileNotFoundError:
        logging.error(f"'dustysgdisc.in' not found for simulation {sim_id}. Skipping.")
        return False

    # Step 6: Generate the submission script
    submission_script = f"""#!/bin/bash
#SBATCH --job-name=ppd_{sim_id:04d}                             # Job name
#SBATCH --partition=batch                                       # Partition name
#SBATCH --ntasks=1                                              # 1 task (process)
#SBATCH --cpus-per-task=20                                      # CPU core count per task
#SBATCH --mem=10G                                               # Memory per node
#SBATCH --time=6-23:59:59                                       # Time limit (days-hours:minutes:seconds)
#SBATCH --output={output_dir}/ppd_{sim_id:04d}_%j.out            # Standard output log
#SBATCH --mail-user=adm61595@uga.edu                            # Replace with your email
#SBATCH --mail-type=FAIL                                        # Mail events (BEGIN, END, FAIL, ALL)

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

    return True

# ==========
# ++ MAIN ++
# ==========

def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    # Command Line Arguments
    parser = argparse.ArgumentParser(description="Generate Protoplanetary Disk Simulations")

    # Positional argument for number of simulations
    parser.add_argument('n_sims', type=int, help='Number of simulations to generate')

    # Optional argument for output directory
    parser.add_argument('-d', '--output_dir', type=str, default='/scratch/0_sink',
                        help='Output directory for simulations (default: /scratch/0_sink)')

    args = parser.parse_args()

    # Assign arguments to variables
    n_sims = args.n_sims
    output_dir = Path(args.output_dir).expanduser()

    logging.info(f"Number of simulations to generate: {n_sims}")
    logging.info(f"Output directory: {output_dir}")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generators and managers
    generator = PhysicalPPDGenerator(seed=42) # For reproducibility

    # Initialize PHANTOMFileManager with the setup template only
    current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    setup_template_path = current_dir / 'dustysgdisc.setup'

    try:
        file_manager = PHANTOMFileManager(str(setup_template_path))
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Create a submit_all script
    submit_all_path = output_dir / 'submit_all.sh'
    with open(submit_all_path, 'w') as f:
        f.write('#!/bin/bash\n')

    # Make the submit_all.sh executable
    submit_all_path.chmod(submit_all_path.stat().st_mode | 0o111)

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
        success = generate_phantom_input(params, sim_dir, i, file_manager)

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
    df.to_csv(output_dir / 'parameter_database.csv', index=False)

    # Summary output
    logging.info(f"\nGenerated {len(param_records)} disc configurations")
    logging.info(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    main()
