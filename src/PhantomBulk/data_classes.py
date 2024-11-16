#!/usr/bin/env python3
# src/PhantomBulk/data_classes.py

from dataclasses import dataclass, field
from typing import List

@dataclass
class PlanetParameters:
    mass: float              # Planet mass in Jupiter masses
    radius: float            # Orbital radius in AU
    inclination: float       # Orbital inclination in degrees
    accr_radius: float       # Accretion radius in Hill radii
    j2_moment: float         # J2 moment (oblateness)

@dataclass
class PPDParameters:
    m1: float                 # Stellar mass in solar masses
    accr1: float              # Stellar accretion radius
    J2_body1: float           # Stellar quadrupole moment
    disc_m: float             # Disc mass
    Sigma0: float             # Surface density
    R_in: float               # Inner radius in AU
    R_out: float              # Outer radius in AU
    R_ref: float              # Reference Radius
    H_R: float                # Aspect ratio
    pindex: float             # Surface density power-law index
    qindex: float             # Temperature power-law index
    dust_to_gas: float        # Dust-to-gas mass ratio
    grainsize: float          # Grain size in cm
    graindens: float          # Grain density in g/cm^3
    beta_cool: float          # Cooling parameter
    planets: List[PlanetParameters] = field(default_factory=list)  # List of planets

@dataclass
class SimulationConstraints:
    """Constraints from observational data and physical limits"""
    min_star_mass: float = 0.08  # Brown dwarf limit (M_sun)
    max_star_mass: float = 8.0   # Herbig Ae/Be limit (M_sun)
    min_disk_mass: float = 1e-4  # Minimum detectable disk mass (M_sun)
    max_disk_mass: float = 0.1   # Gravitational stability limit
    min_r_in: float = 0.01        # Dust sublimation radius (AU)
    max_r_out: float = 1000.0     # Typical outer disk limit (AU)
    min_aspect: float = 0.01      # Minimum stable H/R
    max_aspect: float = 0.25      # Maximum physical H/R
    min_dtg: float = 0.001         # Depleted dust-to-gas
    max_dtg: float = 0.1           # Enhanced dust-to-gas
    min_grain: float = 1e-5        # Minimum grain size (cm)
    max_grain: float = 1.0         # Maximum grain size (cm)