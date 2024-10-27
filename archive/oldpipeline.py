import os
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class PlanetParameters:
    mass: float           # Jupiter masses
    radius: float        # AU
    inclination: float   # degrees
    accr_radius: float   # Hill radius fraction
    j2_moment: float     # Quadrupole moment

@dataclass
class PPDParameters:
    m1: float           # Star mass (solar masses)
    accr1: float        # Star accretion radius (AU)
    disc_m: float       # Disc mass (solar masses)
    R_in: float         # Inner radius (AU)
    R_out: float        # Outer radius (AU)
    H_R: float         # Aspect ratio
    pindex: float      # Surface density power-law index
    qindex: float      # Temperature power-law index
    dust_to_gas: float # Dust-to-gas ratio
    grainsize: float   # Maximum grain size (cm)
    graindens: float   # Grain density (g/cm³)
    beta_cool: float   # Cooling parameter
    planets: List[PlanetParameters]
    
    def validate(self) -> bool:
        """Validate physical consistency of parameters"""
        try:
            # Basic range checks
            if not (0.1 <= self.m1 <= 5.0): return False
            if not (0.0001 <= self.disc_m <= 0.1): return False
            if not (0.01 <= self.R_in <= 10): return False
            if not (10 <= self.R_out <= 1000): return False
            if not (self.R_in < self.R_out): return False
            if not (0.01 <= self.H_R <= 0.2): return False
            
            # Planet checks
            for planet in self.planets:
                if not (0.1 <= planet.mass <= 1000): return False  # Jupiter masses
                if not (self.R_in < planet.radius < self.R_out): return False
                if not (0 <= planet.inclination <= 40): return False
            
            return True
        except:
            return False

def generate_phantom_input(params: PPDParameters, output_dir: Path, sim_id: int):
    """Generate Phantom input file"""
    template = """# Phantom input file for PPD simulation {sim_id}
# Generated automatically

# Units
au=1.496d13
solarm=1.989d33
yearf=3.156d7

# Stellar parameters
mass1={m1}    # Solar masses
accr1={accr1} # AU

# Disc parameters
disc_m={disc_m}  # Solar masses
R_in={R_in}      # AU
R_out={R_out}    # AU
H_R={H_R}        # Aspect ratio
pindex={pindex}  # Surface density index
qindex={qindex}  # Temperature index

# Dust parameters
dust_to_gas={dust_to_gas}
grainsize={grainsize}
graindens={graindens}

# Cooling
beta_cool={beta_cool}

# Planet parameters
nplanets={n_planets}
"""
    
    # Add planet sections if present
    planet_template = """
# Planet {i}
mass_p{i}={mass}
r_p{i}={radius}
incl_p{i}={incl}
accr_p{i}={accr}
j2_p{i}={j2}
"""
    
    input_str = template.format(
        sim_id=sim_id,
        **asdict(params),
        n_planets=len(params.planets)
    )
    
    for i, planet in enumerate(params.planets, 1):
        input_str += planet_template.format(
            i=i,
            mass=planet.mass,
            radius=planet.radius,
            incl=planet.inclination,
            accr=planet.accr_radius,
            j2=planet.j2_moment
        )
    
    # Write input file
    input_path = output_dir / f'disc_{sim_id}.in'
    with open(input_path, 'w') as f:
        f.write(input_str)
    
    # Generate submission script
    submit_script = f"""#!/bin/bash
#SBATCH --job-name=ppd_{sim_id}
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load phantom

cd {output_dir}
phantom disc_{sim_id}.in
"""
    
    script_path = output_dir / f'run_{sim_id}.sh'
    with open(script_path, 'w') as f:
        f.write(submit_script)

def main():
    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_sims', type=int, help='Number of simulations to generate')
    args = parser.parse_args()
    
    # Setup directories
    base_dir = Path('phantom_runs')
    base_dir.mkdir(exist_ok=True)
    
    # Create submit_all script
    with open(base_dir / 'submit_all.sh', 'w') as f:
        f.write('#!/bin/bash\n')
    
    # Generate configurations
    generator = PhysicalPPDGenerator(seed=42)
    param_records = []
    
    for i in range(args.n_sims):
        # Generate physically consistent parameters
        params = generator.generate_single_ppd()
        
        # Generate Phantom input and job script
        generate_phantom_input(params, base_dir, i)
        
        # Add to submit_all script
        with open(base_dir / 'submit_all.sh', 'a') as f:
            f.write(f'sbatch run_{i}.sh\n')

        # Record parameters
        param_dict = asdict(params)
        param_dict['simulation_id'] = i
        if params.planets:
            param_dict['planets'] = str([asdict(planet) for planet in params.planets])
        else:
            param_dict['planets'] = ''
        param_records.append(param_dict)
    
    # Save parameter database
    df = pd.DataFrame(param_records)
    df.to_csv(base_dir / 'parameter_database.csv', index=False)
    
    print(f"Generated {args.n_sims} disc configurations")
    print(f"Files saved in: {base_dir}")

if __name__ == "__main__":
    main()
