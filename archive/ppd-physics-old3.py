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
    R_in: float          # Inner radius (AU)
    R_out: float         # Outer radius (AU)
    H_R: float           # Aspect ratio at reference radius
    pindex: float        # Surface density power law index
    qindex: float        # Sound speed power law index

    # Dust parameters
    dust_to_gas: float   # Dust to gas ratio
    grainsize: float     # Grain size in cm
    graindens: float     # Grain density in g/cm³

    # Cooling parameters
    beta_cool: float     # Beta cooling parameter

    # Planet parameters
    planets: List[PlanetParameters] = field(default_factory=list)

    def validate(self) -> bool:
        """Check if parameters are within physical limits"""
        try:
            # Basic physical constraints - now more relaxed
            assert self.m1 > 0, "Star mass must be positive"
            assert self.disc_m > 0, "Disc mass must be positive"
            assert self.R_in < self.R_out, "Inner radius must be less than outer radius"
            assert self.dust_to_gas > 0, "Dust-to-gas ratio must be positive"

            # Toomre Q criterion - relaxed but still checking for extreme instability
            Q_approx = (self.H_R) * (self.m1/self.disc_m) * (1/self.H_R)
            assert Q_approx > 0.5, "Disc is extremely gravitationally unstable"

            # Validate planet parameters
            for planet in self.planets:
                assert planet.mass > 0, "Planet mass must be positive"
                assert planet.radius > self.R_in, "Planet must be outside inner disc radius"
                assert planet.radius < self.R_out, "Planet must be inside outer disc radius"

            return True
        except AssertionError as e:
            print(f"Validation failed: {e}")
            return False

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
            self.temp_params['T0_mean'] + 0.5 * np.log10(stellar_mass),
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
        log_radius = (np.log10(disk_mass) * self.disk_params['size_mass_slope'] + 
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
        max_planets = min(4, int(disk_mass/0.001))  # More massive disks can support more planets
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
            # Generate stellar mass
            stellar_mass = self.generate_stellar_mass()
            
            # Temperature structure
            T0, q = self.compute_temperature_structure(stellar_mass)
            
            # Disk structure
            disk_mass, R_out, R_in, H_R = self.compute_disk_structure(
                stellar_mass, T0, q
            )
            
            # Dust properties
            dust_to_gas, grain_size, grain_dens = self.generate_dust_properties(disk_mass)
            
            # Surface density profile (based on disk mass distribution)
            pindex = -2.0 * q - 0.5 + np.random.normal(0, 0.2)
            
            # Cooling parameter (based on opacity and temperature)
            beta_cool = 3 + 7 * (T0/280)**0.5 * np.exp(np.random.normal(0, 0.3))
            
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
                graindens=grain_dens,
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

    def __init__(self, base_setup_file: str, base_in_file: str):
        self.base_setup = self.read_file(base_setup_file)
        self.base_in = self.read_file(base_in_file)

    @staticmethod
    def read_file(filename: str) -> List[str]:
        with open(filename, 'r') as f:
            return f.readlines()

    def create_simulation_files(self, params: PPDParameters,
                              setup_file: str, in_file: str):
        """Create new .setup and .in files with given parameters"""
        # Modify .setup file
        setup_contents = self.base_setup.copy()
        setup_modifications = {
            'beta_cool': f"           beta_cool = {params.beta_cool:>11.3f}",
            'grainsize': f"           grainsize = {params.grainsize:>11.3f}",
            'graindens': f"           graindens = {params.graindens:>11.3f}"
        }

        # Add planet configurations
        if params.planets:
            setup_modifications['nplanets'] = f"            nplanets = {len(params.planets):>11d}"
            for i, planet in enumerate(params.planets, 1):
                prefix = f"planet:{i}"
                setup_modifications.update({
                    f'mplanet{i}': f"            mplanet{i} = {planet.mass:>11.3f}",
                    f'rplanet{i}': f"            rplanet{i} = {planet.radius:>11.3f}",
                    f'inclplanet{i}': f"         inclplanet{i} = {planet.inclination:>11.3f}",
                    f'accrplanet{i}': f"         accrplanet{i} = {planet.accr_radius:>11.3f}",
                    f'J2_planet{i}': f"          J2_planet{i} = {planet.j2_moment:>11.3f}"
                })

        # Modify .in file
        in_contents = self.base_in.copy()
        in_modifications = {
            'm1': f"                  m1 = {params.m1:>11.3f}",
            'accr1': f"               accr1 = {params.accr1:>11.3f}",
            'disc_m': f"              disc_m = {params.disc_m:>11.3f}",
            'R_in': f"                R_in = {params.R_in:>11.3f}",
            'R_out': f"               R_out = {params.R_out:>11.3f}",
            'H_R': f"                 H_R = {params.H_R:>11.3f}",
            'pindex': f"              pindex = {params.pindex:>11.3f}",
            'qindex': f"              qindex = {params.qindex:>11.3f}",
            'dust_to_gas': f"         dust_to_gas = {params.dust_to_gas:>11.3f}"
        }

        # Apply modifications
        self._modify_file_contents(setup_contents, setup_modifications)
        self._modify_file_contents(in_contents, in_modifications)

        # Write new files
        self._write_file(setup_file, setup_contents)
        self._write_file(in_file, in_contents)

    @staticmethod
    def _modify_file_contents(contents: List[str], modifications: Dict[str, str]):
        """Modify file contents based on parameter dictionary"""
        for i, line in enumerate(contents):
            for key, new_line in modifications.items():
                if key in line and '=' in line:
                    contents[i] = new_line + '\n'
                    break

    @staticmethod
    def _write_file(filename: str, contents: List[str]):
        """Write contents to file"""
        with open(filename, 'w') as f:
            f.writelines(contents)

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
            for i, planet in enumerate(eval(row['planets'])):
                planet_data.append({
                    'simulation_id': sim_id,
                    'planet_number': i + 1,
                    'mass': planet['mass'],
                    'radius': planet['radius'],
                    'inclination': planet['inclination']
                })

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

# ==========
# ++ MAIN ++
# ==========

def main():
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('n_sims', type=int, help='Number of simulations to generate')
    args = parser.parse_args()

    # Setup base directory
    base_dir = Path('phantom_runs')
    base_dir.mkdir(exist_ok=True)
    
    # Initialize generators and managers
    generator = PhysicalPPDGenerator(seed=42)  # Assuming this covers physical consistency
    file_manager = PHANTOMFileManager('dustysgdisc.setup', 'dustysgdisc.in')

    # Create a submit_all script
    submit_all_path = base_dir / 'submit_all.sh'
    with open(submit_all_path, 'w') as f:
        f.write('#!/bin/bash\n')
    
    # Generate parameters and setup/input files
    param_records = []
    for i in range(args.n_sims):
        # Generate physically consistent parameters
        params = generator.generate_single_ppd()
        
        # Set up simulation directory
        sim_dir = base_dir / f'sim_{i:04d}'
        sim_dir.mkdir(exist_ok=True)
        
        # Create setup and input files
        setup_file = sim_dir / 'dustysgdisc.setup'
        in_file = sim_dir / 'dustysgdisc.in'
        file_manager.create_simulation_files(params, setup_file, in_file)
        
        # Generate Phantom input and job script
        # Initialize PHANTOM Makefile and compile
        subprocess.run(['~/phantom/scripts/writemake.sh', 'dustysgdisc', '>', str(sim_dir / 'Makefile')], shell=True, cwd=sim_dir)
        subprocess.run(['make'], cwd=sim_dir)
        subprocess.run(['make', 'setup'], cwd=sim_dir)

        # Run PHANTOM setup to create initial dump files
        subprocess.run(['./phantomsetup', f'sim_{i:04d}'], cwd=sim_dir)

        generate_phantom_input(params, base_dir, i)
        
        # Add job to submit_all script
        with open(submit_all_path, 'a') as f:
            f.write(f'cd {sim_dir} && ./phantom {in_file.name}\n')

        # Record parameters in a dictionary
        param_dict = asdict(params)
        param_dict['simulation_id'] = i
        param_dict['planets'] = str([asdict(planet) for planet in params.planets]) if params.planets else ''
        param_records.append(param_dict)

    # Save parameters to CSV
    df = pd.DataFrame(param_records)
    df.to_csv(base_dir / 'parameter_database.csv', index=False)

    # Create interactive visualizations if desired
    create_parameter_visualizations(df, base_dir)

    # Summary output
    print(f"Generated {args.n_sims} disc configurations")
    print(f"Total number of planets: {sum(len(eval(row['planets'])) for row in df['planets'] if row)}")
    print(f"Files saved in: {base_dir}")
    print("Interactive visualizations have been created as HTML files in the output directory")

if __name__ == "__main__":
    main()
