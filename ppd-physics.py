import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import asdict
import argparse
import subprocess

# ===========================
# ++ SIMULATION PARAMETERS ++
# ===========================

@dataclass
class PlanetParameters:
    """Class to hold parameters for a single planet"""
    mass: float           # Planet mass (Jupiter masses)
    radius: float         # Orbital radius (AU)
    inclination: float    # Orbital inclination (degrees)
    accr_radius: float    # Accretion radius (Hill radii)
    j2_moment: float      # J2 moment (oblateness)

@dataclass
class PPDParameters:
    """Class to hold and validate PPD parameters"""
    # Stellar parameters
    m1: float              # Central star mass (solar masses)
    accr1: float          # Star accretion radius

    # Disc parameters
    disc_m: float         # Disc mass (solar masses)
    R_in: float           # Inner radius (AU)
    R_out: float          # Outer radius (AU)
    H_R: float            # Aspect ratio at reference radius
    pindex: float         # Surface density power law index
    qindex: float         # Sound speed power law index

    # Dust parameters
    dust_to_gas: float    # Dust to gas ratio
    grainsize: float      # Grain size in cm
    graindens: float      # Grain density in g/cm³

    # Cooling parameters
    beta_cool: float      # Beta cooling parameter

    # Planet parameters
    planets: List[PlanetParameters] = field(default_factory=list)

    def validate(self) -> bool:
        """Check if parameters are within physical limits, including Toomre Q"""
        try:
            # Basic physical constraints
            assert self.m1 > 0, "Star mass must be positive"
            assert self.disc_m > 0, "Disc mass must be positive"
            assert self.R_in < self.R_out, "Inner radius must be less than outer radius"
            assert self.dust_to_gas > 0, "Dust-to-gas ratio must be positive"

            # Physical constants in cgs units
            G_cgs = 6.67430e-8       # Gravitational constant (cm³ g⁻¹ s⁻²)
            M_sun_cgs = 1.98847e33   # Solar mass (g)
            AU_cgs = 1.495978707e13  # Astronomical Unit (cm)

            # Convert stellar mass and radii to cgs units
            M_star = self.m1 * M_sun_cgs      # Stellar mass (g)
            R_ref = self.R_in * AU_cgs        # Reference radius (cm)

            # Calculate surface density Σ at R_ref
            if self.pindex == 2:
                raise ValueError("pindex = 2 is not supported due to singularity in surface density calculation.")
            else:
                factor = (2 - self.pindex) / (2 * np.pi * self.R_in**self.pindex * ( (self.R_out / self.R_in)**(2 - self.pindex) - 1 ))
                Sigma0 = (self.disc_m * M_sun_cgs) * factor / (AU_cgs**self.pindex)  # Σ0 in g/cm²

            # Surface density at R_ref
            Sigma_ref = Sigma0  # Since (R/R_ref)^(-p) = 1 at R = R_ref

            # Calculate sound speed c_s at R_ref
            v_orb = np.sqrt(G_cgs * M_star / R_ref)  # Orbital velocity at R_ref (cm/s)
            c_s = self.H_R * v_orb                   # Sound speed (cm/s)

            # Calculate epicyclic frequency κ
            kappa = np.sqrt(G_cgs * M_star / R_ref**3)  # Epicyclic frequency (rad/s)

            # Calculate Toomre Q parameter
            Q = (c_s * kappa) / (np.pi * G_cgs * Sigma_ref)

            # Ensure Toomre Q is above the critical threshold
            Q_threshold = 1.5
            assert Q > Q_threshold, f"Toomre Q ({Q:.2f}) is below the critical threshold ({Q_threshold}) indicating gravitational instability."

            # Validate planet parameters
            for planet in self.planets:
                assert planet.mass > 0, "Planet mass must be positive"
                assert planet.radius > self.R_in, "Planet must be outside inner disc radius"
                assert planet.radius < self.R_out, "Planet must be inside outer disc radius"
                assert 0 <= planet.inclination <= 30, "Planet inclination must be between 0 and 30 degrees"

            # Validate number of planets
            assert 0 <= len(self.planets) <= 6, "Number of planets must be between 0 and 6"

            return True
        except AssertionError as e:
            print(f"Validation failed: {e}")
            return False
        except ValueError as e:
            print(f"Validation failed: {e}")
            return False

    def planet_configurations(self) -> str:
        """Generate the planet configuration string for the .setup file"""
        planet_config = ""
        for i, planet in enumerate(self.planets, 1):
            planet_config += f"""
# Planet {i}
mplanet{i} =       {planet.mass:.3f}    ! planet mass (Jupiter masses)
rplanet{i} =       {planet.radius:.3f}      ! Orbital radius (AU)
inclplanet{i} =    {planet.inclination:.3f}  ! Orbital inclination (degrees)
accrplanet{i} =    {planet.accr_radius:.3f}  ! Accretion radius (Hill radii)
J2_planet{i} =     {planet.j2_moment:.3f}    ! J2 moment (oblateness)
"""
        return planet_config

# ========================
# ++ SAMPLING FUNCTION ++
# ========================

def sample_parameter(core_range: Tuple[float, float],
                    tail_range: Tuple[float, float],
                    tail_probability: float = 0.05) -> float:
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

# ===========================
# ++ PPD GENERATOR CLASS ++
# ===========================

class PhysicalPPDGenerator:
    """Generate PPD parameters with physical correlations"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            
        # Load empirical distributions from surveys
        self.load_survey_distributions()

    def load_survey_distributions(self):
        """Load empirical distributions from astronomical surveys"""
        # Stellar mass distribution (Kroupa IMF)
        self.imf_alpha = -2.3  # High-mass slope
        self.mass_break = 0.5  # Solar masses
        
        # Disk properties from surveys (e.g., Andrews et al. 2010, Ansdell et al. 2016)
        self.disk_params = {
            'mass_ratio_mean': np.log(0.01),  # Mean log disk-to-star mass ratio
            'mass_ratio_std': 0.5,            # Scatter in log mass ratio
            'size_mass_slope': 0.5,           # Power-law index for R_out vs M_disk
            'size_mass_scatter': 0.2          # Scatter in size-mass relation
        }
        
        # Temperature profile parameters
        self.temp_params = {
            'T0_mean': np.log(300),  # Mean log temperature at 1 AU
            'T0_std': 0.2,           # Scatter in log temperature
            'q_mean': -0.5,          # Mean temperature power-law index
            'q_std': 0.1             # Scatter in temperature index
        }
        
        # Define core and tail ranges for each parameter
        self.parameter_ranges = {
            'm1': {
                'core': (0.2, 2.0),
                'tail': (0.1, 5.0)
            },
            'accr1': {
                'core': (0.05, 0.2),
                'tail': (0.02, 0.5)
            },
            'disc_m': {
                'core': (0.005, 0.05),
                'tail': (0.0001, 0.1)
            },
            'R_in': {
                'core': (0.1, 0.5),
                'tail': (0.05, 1.0)
            },
            'R_out': {
                'core': (50, 100),
                'tail': (20, 200)
            },
            'H_R': {
                'core': (0.08, 0.15),
                'tail': (0.05, 0.25)
            },
            'pindex': {
                'core': (-1.0, -0.5),
                'tail': (-1.5, -0.3)
            },
            'qindex': {
                'core': (-0.5, -0.3),
                'tail': (-0.7, -0.2)
            },
            'dust_to_gas': {
                'core': (0.01, 0.05),
                'tail': (0.005, 0.1)
            },
            'grainsize': {
                'core': (0.001, 0.1),
                'tail': (1e-4, 1.0)
            },
            'graindens': {
                'core': (2.5, 3.5),
                'tail': (1.5, 4.5)
            },
            'beta_cool': {
                'core': (3, 10),
                'tail': (1, 20)
            }
        }

    def generate_stellar_mass(self) -> float:
        """Generate stellar mass following the IMF"""
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
        """Compute disk temperature structure"""
        # Temperature at 1 AU (influenced by stellar mass)
        log_T0 = np.random.normal(
            self.temp_params['T0_mean'] + 0.5 * np.log(stellar_mass),
            self.temp_params['T0_std']
        )
        T0 = np.exp(log_T0)
        
        # Temperature power-law index
        q = np.random.normal(
            self.temp_params['q_mean'],
            self.temp_params['q_std']
        )
        
        return T0, q

    def compute_disk_structure(self, stellar_mass: float, T0: float, q: float) -> tuple:
        """Compute physically consistent disk structure"""
        # Disk mass based on stellar mass with scatter
        log_mass_ratio = np.random.normal(
            self.disk_params['mass_ratio_mean'],
            self.disk_params['mass_ratio_std']
        )
        disk_mass = stellar_mass * np.exp(log_mass_ratio)
        
        # Outer radius based on disk mass
        log_radius = (np.log(disk_mass) * self.disk_params['size_mass_slope'] + 
                     np.random.normal(0, self.disk_params['size_mass_scatter']))
        R_out = 10**log_radius * 100  # Convert to AU
        
        # Inner radius based on dust sublimation
        R_in = 0.1 * (stellar_mass/1.0)**0.5  # Approximate dust sublimation radius
        
        # Aspect ratio from temperature structure
        H_R = np.sqrt(T0/280) * (stellar_mass/1.0)**(-0.5)
        
        return disk_mass, R_out, R_in, H_R

    def generate_dust_properties(self, disk_mass: float) -> tuple:
        """Generate dust properties considering disk mass"""
        # Dust-to-gas ratio (higher in more massive disks)
        dust_to_gas = 0.01 * (disk_mass/0.01)**0.2 * np.exp(np.random.normal(0, 0.3))
        
        # Grain size distribution
        max_grain_size = 10**np.random.uniform(-4, 0)  # cm
        grain_density = 3.0  # g/cm³ (typical silicate density)
        
        return dust_to_gas, max_grain_size, grain_density

    def generate_planet_system(self, stellar_mass: float, disk_mass: float,
                               R_in: float, R_out: float) -> List[PlanetParameters]:
        """Generate physically consistent planetary system"""
        # Determine number of planets with limits
        max_planets = min(6, int(disk_mass/0.001))  # More massive disks can support more planets
        n_planets = np.random.randint(0, max_planets + 1)
        
        if n_planets == 0:
            return []
        
        # Generate planet locations using power-law spacing
        available_radii = np.logspace(np.log10(R_in*1.5), np.log10(R_out*0.8), 100)
        planet_radii = np.sort(np.random.choice(available_radii, n_planets, replace=False))
        
        planets = []
        for radius in planet_radii:
            # Planet mass based on disk mass and location
            max_mass = min(
                disk_mass * 318,  # Convert to Jupiter masses
                (disk_mass/0.01) * (radius/30)**(-1.5)  # Mass decreases with radius
            )
            mass = np.random.uniform(0.1, max_mass)
            
            # Inclination dependent on system mass
            max_incl = 10 * (stellar_mass/disk_mass)**0.2
            incl = np.random.rayleigh(max_incl/3)
            incl = min(incl, 30)  # Enforce inclination limit
            
            planet = PlanetParameters(
                mass=mass,
                radius=radius,
                inclination=incl,
                accr_radius=0.3,  # Hill radius fraction
                j2_moment=0.0     # Simplified to spherical planets
            )
            planets.append(planet)
            
        return planets

    def generate_single_ppd(self) -> PPDParameters:
        """Generate a single physically consistent PPD"""
        while True:
            # Generate stellar mass with core and tail ranges
            stellar_mass = sample_parameter(
                self.parameter_ranges['m1']['core'],
                self.parameter_ranges['m1']['tail']
            )
            
            # Temperature structure
            T0, q = self.compute_temperature_structure(stellar_mass)
            
            # Disk structure with core and tail ranges
            disk_mass = sample_parameter(
                self.parameter_ranges['disc_m']['core'],
                self.parameter_ranges['disc_m']['tail']
            )
            R_out = sample_parameter(
                self.parameter_ranges['R_out']['core'],
                self.parameter_ranges['R_out']['tail']
            )
            R_in = sample_parameter(
                self.parameter_ranges['R_in']['core'],
                self.parameter_ranges['R_in']['tail']
            )
            H_R = sample_parameter(
                self.parameter_ranges['H_R']['core'],
                self.parameter_ranges['H_R']['tail']
            )
            
            # Dust properties with core and tail ranges
            dust_to_gas, grain_size, graindens = self.generate_dust_properties(disk_mass)
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
            
            # Surface density profile (based on disk mass distribution)
            pindex = sample_parameter(
                self.parameter_ranges['pindex']['core'],
                self.parameter_ranges['pindex']['tail']
            )
            
            # Cooling parameter (based on opacity and temperature)
            beta_cool = sample_parameter(
                self.parameter_ranges['beta_cool']['core'],
                self.parameter_ranges['beta_cool']['tail']
            )
            
            # Generate planetary system
            planets = self.generate_planet_system(
                stellar_mass, disk_mass, R_in, R_out
            )
            
            # Create parameter object
            params = PPDParameters(
                m1=stellar_mass,
                accr1=R_in/2,
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
                planets=planets
            )
            
            if params.validate():
                return params

    def generate_parameter_set(self, n_discs: int) -> List[PPDParameters]:
        """Generate parameters for multiple discs"""
        return [self.generate_single_ppd() for _ in range(n_discs)]

# ==================
# ++ FILE MANAGER ++
# ==================

class PHANTOMFileManager:
    """Handle PHANTOM input file generation and modification"""

    def __init__(self, setup_template_path: str, in_template_path: str):
        self.setup_template = self.read_file(setup_template_path)
        self.in_template = self.read_file(in_template_path)

    @staticmethod
    def read_file(filename: str) -> str:
        with open(filename, 'r') as f:
            return f.read()

    def create_simulation_files(self, params: PPDParameters, output_dir: Path, sim_id: int):
        """Create new .setup and .in files with given parameters"""
        # Replace placeholders in .setup file
        setup_content = self.setup_template
        setup_content = setup_content.replace("{{NUM_PLANETS}}", str(len(params.planets)))
        setup_content = setup_content.replace("{{PLANET_CONFIGURATIONS}}", params.planet_configurations())

        # Replace placeholders in .in file
        in_content = self.in_template
        in_content = in_content.replace("{{SIM_ID:02d}}", f"{sim_id:02d}")
        in_content = in_content.replace("{{SIM_ID:05d}}", f"{sim_id:05d}")
        in_content = in_content.format(
            beta_cool=params.beta_cool,
            grainsize=params.grainsize,
            graindens=params.graindens
        )

        # Write the .setup file
        setup_file = output_dir / 'dustysgdisc.setup'
        with open(setup_file, 'w') as f:
            f.write(setup_content)

        # Write the .in file
        in_file = output_dir / 'dustysgdisc.in'
        with open(in_file, 'w') as f:
            f.write(in_content)

# ==============================
# ++ PARAMETER VISUALIZATIONS ++
# ==============================

def create_parameter_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create interactive 2D and 3D visualizations using plotly"""

    # 2D scatter plot: Disc mass vs Stellar mass with H/R as color
    fig_2d = px.scatter(df,
                       x='m1',
                       y='disc_m',
                       color='H_R',
                       hover_data=['R_in', 'R_out', 'beta_cool'],
                       title='Disc Parameters Distribution',
                       labels={'m1': 'Stellar Mass (M☉)',
                              'disc_m': 'Disc Mass (M☉)',
                              'H_R': 'H/R'})
    fig_2d.write_html(output_dir / 'parameter_distribution_2d.html')

    # 3D scatter plot: Stellar mass, disc mass, and outer radius
    fig_3d = px.scatter_3d(df,
                          x='m1',
                          y='disc_m',
                          z='R_out',
                          color='beta_cool',
                          hover_data=['H_R', 'dust_to_gas'],
                          title='3D Parameter Space',
                          labels={'m1': 'Stellar Mass (M☉)',
                                 'disc_m': 'Disc Mass (M☉)',
                                 'R_out': 'Outer Radius (AU)',
                                 'beta_cool': 'β cooling'})
    fig_3d.write_html(output_dir / 'parameter_distribution_3d.html')

    # Create planet distribution visualization if planets exist
    planet_data = []
    for idx, row in df.iterrows():
        sim_id = row['simulation_id']
        if 'planets' in row and row['planets']:
            try:
                planets = eval(row['planets'])  # Safe if data is controlled
                for i, planet in enumerate(planets):
                    planet_data.append({
                        'simulation_id': sim_id,
                        'planet_number': i + 1,
                        'mass': planet['mass'],
                        'radius': planet['radius'],
                        'inclination': planet['inclination']
                    })
            except Exception as e:
                print(f"Error parsing planets for simulation {sim_id}: {e}")

    if planet_data:
        planet_df = pd.DataFrame(planet_data)
        fig_planets = px.scatter(planet_df,
                               x='radius',
                               y='mass',
                               color='inclination',
                               title='Planet Distribution',
                               labels={'radius': 'Orbital Radius (AU)',
                                     'mass': 'Planet Mass (M_J)',
                                     'inclination': 'Inclination (deg)'})
        fig_planets.write_html(output_dir / 'planet_distribution.html')

# ==============================
# ++ GENERATE PHANTOM INPUT ++
# ==============================

def generate_phantom_input(params: PPDParameters, output_dir: Path, sim_id: int, file_manager: PHANTOMFileManager):
    """Generate PHANTOM setup and input files, and the submission script for a simulation."""
    # Use the file manager to create .setup and .in files
    file_manager.create_simulation_files(params, output_dir, sim_id)

    # ===============================
    # ++ SUBMISSION SCRIPT (.sh) ++
    # ===============================

    # SBATCH directives and environment settings based on Sapelo2
    submission_script = f"""#!/bin/bash
# ====================
# PARAMETERS FOR JOB |
# ====================
#SBATCH --job-name=ppd_{sim_id:04d}                # Job name
#SBATCH --partition=batch                          # Partition name (batch, highmem_p, or gpu_p)
#SBATCH --ntasks=1                                 # 1 task (process)
#SBATCH --cpus-per-task=28                         # CPU core count per task
#SBATCH --mem=32G                                  # Memory per node
#SBATCH --time=6-23:59:59                          # Time limit (days-hours:minutes:seconds)
#SBATCH --output=ppd_{sim_id:04d}_%j.out           # Standard output log
#SBATCH --mail-user=your_email@example.com          # Replace with your email
#SBATCH --mail-type=FAIL                           # Mail events (BEGIN, END, FAIL, ALL)

# Load necessary modules
module load phantom

# Set environment variables
source ~/.bashrc                                # Ensure ~/.bashrc is sourced
export SYSTEM=gfortran
ulimit -s unlimited
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=28
export OMP_STACKSIZE=1024M

# Change to the simulation directory
cd {output_dir}

# Run PHANTOM with the input file
phantom dustysgdisc.in
"""

    # Write the submission script
    submit_script_path = output_dir / f'run_{sim_id}.sh'
    with open(submit_script_path, 'w') as f:
        f.write(submission_script)

    # Make the submission script executable
    submit_script_path.chmod(submit_script_path.stat().st_mode | 0o111)

# ==========
# ++ MAIN ++
# ==========

def main():
    # Command Line Arguments
    parser = argparse.ArgumentParser(description="Generate Protoplanetary Disk Simulations")
    parser.add_argument('n_sims', type=int, help='Number of simulations to generate')
    args = parser.parse_args()

    # Setup base directory
    base_dir = Path('phantom_runs')
    base_dir.mkdir(exist_ok=True)

    # Initialize generators and managers
    generator = PhysicalPPDGenerator(seed=42)  # Ensures reproducibility
    # Initialize PHANTOMFileManager with paths to your template files
    file_manager = PHANTOMFileManager('dustysgdisc.setup', 'dustysgdisc.in')

    # Create a submit_all script
    submit_all_path = base_dir / 'submit_all.sh'
    with open(submit_all_path, 'w') as f:
        f.write('#!/bin/bash\n')

    # Make the submit_all.sh executable
    submit_all_path.chmod(submit_all_path.stat().st_mode | 0o111)

    # Generate parameters and setup/input files
    param_records = []
    for i in range(args.n_sims):
        # Generate physically consistent parameters with core and tail sampling
        params = generator.generate_single_ppd()

        # Set up simulation directory
        sim_dir = base_dir / f'sim_{i:04d}'
        sim_dir.mkdir(exist_ok=True)

        # Generate .setup and .in files, and submission script
        generate_phantom_input(params, sim_dir, i, file_manager)

        # Initialize PHANTOM Makefile and compile
        # Note: Adjust the path to writemake.sh as needed
        writemake_cmd = f'~/phantom/scripts/writemake.sh dustysgdisc > {sim_dir}/Makefile'
        result_writemake = subprocess.run(writemake_cmd, shell=True, cwd=sim_dir)
        if result_writemake.returncode != 0:
            print(f"Error: 'writemake.sh' failed for simulation {i}. Skipping.")
            continue

        # Compile PHANTOM
        result_make = subprocess.run(['make'], cwd=sim_dir)
        if result_make.returncode != 0:
            print(f"Error: 'make' failed for simulation {i}. Skipping.")
            continue

        # Setup PHANTOM
        result_setup = subprocess.run(['make', 'setup'], cwd=sim_dir)
        if result_setup.returncode != 0:
            print(f"Error: 'make setup' failed for simulation {i}. Skipping.")
            continue

        # Run PHANTOM setup to create initial dump files
        result_phantomsetup = subprocess.run(['./phantomsetup', 'dustysgdisc'], cwd=sim_dir)
        if result_phantomsetup.returncode != 0:
            print(f"Error: 'phantomsetup' failed for simulation {i}. Skipping.")
            continue

        # Add job to submit_all.sh
        with open(submit_all_path, 'a') as f:
            f.write(f'sbatch {sim_dir}/run_{i}.sh\n')

        # Record parameters in a dictionary
        param_dict = asdict(params)
        param_dict['simulation_id'] = i
        # Convert planets to list of dictionaries
        param_dict['planets'] = str([asdict(planet) for planet in params.planets]) if params.planets else ''
        param_records.append(param_dict)

        # Optional: Print progress
        if (i+1) % 100 == 0 or (i+1) == args.n_sims:
            print(f"Generated {i+1}/{args.n_sims} simulations")

    # Save parameters to CSV
    df = pd.DataFrame(param_records)
    df.to_csv(base_dir / 'parameter_database.csv', index=False)

    # Create interactive visualizations if desired
    create_parameter_visualizations(df, base_dir)

    # Summary output
    total_planets = sum(len(eval(row['planets'])) for row in df['planets'] if row['planets'])
    print(f"\nGenerated {args.n_sims} disc configurations")
    print(f"Total number of planets: {total_planets}")
    print(f"Files saved in: {base_dir}")
    print("Interactive visualizations have been created as HTML files in the output directory")

if __name__ == "__main__":
    main()
