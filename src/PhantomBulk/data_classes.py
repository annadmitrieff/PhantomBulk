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
    H_R: float                # Aspect ratio
    pindex: float             # Surface density power-law index
    qindex: float             # Temperature power-law index
    dust_to_gas: float        # Dust-to-gas mass ratio
    grainsize: float          # Grain size in cm
    graindens: float          # Grain density in g/cm^3
    beta_cool: float          # Cooling parameter
    T0: float                 # Temperature at reference radius
    planets: List[PlanetParameters] = field(default_factory=list)  # List of planets
