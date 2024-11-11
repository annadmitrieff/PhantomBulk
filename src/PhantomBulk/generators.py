#!/usr/bin/env python3
# src/PhantomBulk/generators.py

import numpy as np
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from .utils import sample_parameter
from .config import Config
from .file_manager import PHANTOMFileManager
from .data_classes import PPDParameters, PlanetParameters  # Import data classes

class PhysicalPPDGenerator:
    """Generate PPD parameters with physical correlations."""
    
    def __init__(self, config: Config):
        """Initialize the generator with configuration parameters."""
        self.config = config
        np.random.seed(self.config.seed)
        
        # Load empirical distributions from surveys
        self.load_survey_distributions()
    
    def load_survey_distributions(self):
        """Load empirical distributions from astronomical surveys."""
        self.parameter_ranges = self.config.parameter_ranges
    
    def compute_temperature_structure(self, stellar_mass: float) -> Tuple[float, float]:
        """
        Compute disc temperature structure.
        
        Parameters:
            stellar_mass (float): Stellar mass in solar masses.
        
        Returns:
            Tuple containing T0 (Kelvin) and q (power-law index).
        """
        # Temperature scaling with stellar mass and radius
        # Ref: Andrews & Williams 2005, ApJ, 631, 1134
        L_star = stellar_mass ** 3.5  # Approximate luminosity scaling
        
        # Temperature at 1 AU with realistic scatter
        log_T0 = np.random.normal(
            np.log(280) + 0.25 * np.log(L_star),
            0.3
        )
        T0 = np.exp(log_T0)
        
        # Temperature power law index
        # Ref: D'Alessio et al. 2001, ApJ, 553, 321
        q = np.random.normal(-0.5, 0.1)
        q = np.clip(q, -0.75, -0.25)  # Physical limits from radiative equilibrium
        
        return T0, q
    
    def compute_disc_structure(
        self, stellar_mass: float, T0: float, q: float
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute physically consistent disc structure.
        
        Parameters:
            stellar_mass (float): Stellar mass in solar masses.
            T0 (float): Temperature at reference radius.
            q (float): Temperature power-law index.
        
        Returns:
            Tuple containing disc_mass, R_out, R_in, Sigma0, pindex.
        """
        # Disc mass as a fraction of stellar mass
        # Ref: Andrews et al. 2013, ApJ, 771, 129
        disc_mass = 0.01 * stellar_mass
        
        # Outer radius sampled from parameter ranges
        R_out = sample_parameter(
            self.parameter_ranges['R_out']['core'],
            self.parameter_ranges['R_out']['tail']
        )
        
        R_in = 1.0  # AU, fixed inner radius
        
        # Surface density at reference radius
        # Ref: D'Alessio et al. 2001, ApJ, 553, 321
        Sigma0 = 1700  # Placeholder value in g/cm^2
        
        # Surface density power-law index
        pindex = 1.5  # Placeholder value
        
        return disc_mass, R_out, R_in, Sigma0, pindex
    
    def compute_aspect_ratio(
        self, T0: float, stellar_mass: float, R_ref: float = 1.0
    ) -> float:
        """
        Compute aspect ratio H/R with physical dependencies.
        
        Parameters:
            T0 (float): Temperature at reference radius.
            stellar_mass (float): Stellar mass in solar masses.
            R_ref (float): Reference radius in AU.
        
        Returns:
            Aspect ratio H/R.
        """
        # Aspect ratio based on temperature and stellar mass
        # Ref: Dullemond & Dominik 2004, A&A, 421, 1075
        k_B = 1.380649e-16  # Boltzmann constant in erg/K
        mu = 2.34           # Mean molecular weight for molecular H2
        m_H = 1.6735575e-24 # Hydrogen mass in g
        
        # Calculate scale height
        # H = c_s / Omega
        # where c_s is sound speed and Omega is orbital frequency
        c_s = np.sqrt(k_B * T0 / (mu * m_H))
        Omega = np.sqrt(1.0 / (R_ref * 1.496e13)**3 * (6.67430e-8 * stellar_mass * 1.98847e33))  # rad/s
        H = c_s / Omega  # cm
        
        # Convert H to AU
        H_AU = H / 1.496e13
        
        # Aspect ratio
        H_R = H_AU / R_ref
        
        return H_R
    
    def generate_planet_system(
        self, stellar_mass: float, disc_mass: float, R_in: float, R_out: float
    ) -> List[PlanetParameters]:
        """
        Generate a physically consistent planetary system.
        
        Parameters:
            stellar_mass (float): Stellar mass in solar masses.
            disc_mass (float): Disc mass.
            R_in (float): Inner radius in AU.
            R_out (float): Outer radius in AU.
        
        Returns:
            List of PlanetParameters.
        """
        # Planet formation efficiency
        # Ref: Mulders et al. 2018, ApJ, 869, L41
        efficiency = np.random.lognormal(mean=np.log(0.1), sigma=1.0)
        max_total_planet_mass = disc_mass * efficiency
        
        # Number of planets based on stellar mass
        # Ref: Zhu & Wu 2018, AJ, 156, 92
        lambda_poisson = 2.0 * (stellar_mass / 1.0) ** 0.5
        n_planets = np.random.poisson(lambda_poisson)
        n_planets = min(n_planets, 8)  # Cap maximum number
        
        if n_planets == 0:
            return []
        
        # Generate planet parameters
        planets = []
        remaining_mass = max_total_planet_mass
        
        # Minimum separation in mutual Hill radii
        # Ref: Pu & Wu 2015, ApJ, 807, 44
        min_separation = 8.0
        
        available_radii = np.logspace(np.log10(R_in * 1.5), np.log10(R_out * 0.8), 1000)
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
                        if abs(np.log10(r / existing_r)) < min_separation / 3:
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
                10.0 * (radius / 30) ** (-1.5)  # Jupiter masses
            )
            
            mass = np.random.power(0.6) * max_mass  # Power law distribution
            remaining_mass -= mass
            
            # Inclination distribution
            # Ref: Xie et al. 2016, PNAS, 113, 11431
            incl = np.random.rayleigh(1.5)
            incl = min(incl, 40)  # Cap maximum inclination
            
            # Accretion radius scaled to Hill radius
            # Ref: Lissauer et al. 2011, ApJS, 197, 8
            hill_radius = radius * (mass / (3 * stellar_mass)) ** (1 / 3)
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
    
    def validate_parameters(self, params: PPDParameters) -> bool:
        """
        Validate generated PPD parameters.
        
        Parameters:
            params (PPDParameters): The generated PPD parameters.
        
        Returns:
            bool: True if valid, False otherwise.
        """
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
        # Assuming H_R_min and H_R_max are defined in config or as constants
        H_R_min = 0.01
        H_R_max = 0.25
        if params.H_R < H_R_min or params.H_R > H_R_max:
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
        T0_min = 10.0  # K
        T0_max = 2000.0  # K
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
    
    def generate_single_ppd(self) -> PPDParameters:
        """
        Generate a single physically consistent PPD.
        
        Returns:
            PPDParameters: The generated PPD parameters.
        
        Raises:
            ValueError: If invalid parameters are generated after multiple attempts.
        """
        max_attempts = 10  # Limit the number of attempts to prevent infinite loops
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            try:
                # Generate stellar mass
                stellar_mass = sample_parameter(
                    self.parameter_ranges['m1']['core'],
                    self.parameter_ranges['m1']['tail']
                )
                logging.debug(f"Generated stellar_mass: {stellar_mass}")
    
                # Compute temperature structure
                T0, q = self.compute_temperature_structure(stellar_mass)
                logging.debug(f"Computed temperature structure: T0={T0}, q={q}")
    
                # Compute disc structure
                disc_mass, R_out, R_in, Sigma0, pindex = self.compute_disc_structure(stellar_mass, T0, q)
                logging.debug(f"Computed disc structure: disc_mass={disc_mass}, R_out={R_out}, R_in={R_in}, Sigma0={Sigma0}, pindex={pindex}")
    
                # Compute aspect ratio
                H_R = self.compute_aspect_ratio(T0, stellar_mass)
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
    
                accr1 = 1.0  # Fixed accretion radius
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
                if self.validate_parameters(params):
                    logging.debug("Parameters validated successfully.")
                    return params
                else:
                    logging.warning("Generated parameters failed validation. Regenerating.")
            except Exception as e:
                logging.error(f"Error during PPD generation: {e}")
                continue
    
        raise ValueError("Failed to generate valid PPD parameters after multiple attempts.")
    
    def generate_parameter_set(self, n_discs: int) -> List[PPDParameters]:
        """
        Generate parameters for multiple discs.
        
        Parameters:
            n_discs (int): Number of discs to generate.
        
        Returns:
            List of PPDParameters.
        """
        return [self.generate_single_ppd() for _ in range(n_discs)]
