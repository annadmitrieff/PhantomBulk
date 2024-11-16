# src/PhantomBulk/generators.py

import numpy as np
import logging
import traceback
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .config import Config
from .data_classes import PPDParameters, PlanetParameters, SimulationConstraints

class PhysicalPPDGenerator:
    """Generate physically self-consistent PPD parameters with advanced physics"""

    def __init__(self, config: Config):
        """Initialize generator with configuration parameters"""
        self.config = config
        self.rng = np.random.RandomState(self.config.seed)
        self._setup_constants()
        self._load_observational_relations()
        self.constraints = self.load_constraints_from_config()

    def load_constraints_from_config(self) -> SimulationConstraints:
        """Load constraints from the configuration"""
        c = self.config.constraints
        constraints = SimulationConstraints(
            min_star_mass=float(c.get('min_star_mass', 0.08)),
            max_star_mass=float(c.get('max_star_mass', 2.5)),
            min_disk_mass=float(c.get('min_disk_mass', 1e-4)),
            max_disk_mass=float(c.get('max_disk_mass', 0.1)),
            min_r_in=float(c.get('min_r_in', 0.01)),
            max_r_out=float(c.get('max_r_out', 1000.0)),
            min_aspect=float(c.get('min_aspect', 0.01)),
            max_aspect=float(c.get('max_aspect', 0.25)),
            min_dtg=float(c.get('min_dtg', 0.001)),
            max_dtg=float(c.get('max_dtg', 0.1)),
            min_grain=float(c.get('min_grain', 1e-5)),
            max_grain=float(c.get('max_grain', 1.0)),
            min_metallicity=float(c.get('min_metallicity', 0.1)),
            max_metallicity=float(c.get('max_metallicity', 2.0)),
            min_B_field=float(c.get('min_B_field', 1e-6)),
            max_B_field=float(c.get('max_B_field', 1e-4))
        )
        return constraints

    def _setup_constants(self):
        """Physical constants in cgs units"""
        self.G = 6.67430e-8             # Gravitational constant
        self.k_B = 1.380649e-16         # Boltzmann constant
        self.m_p = 1.672621898e-24      # Proton mass
        self.sigma_sb = 5.670374419e-5  # Stefan-Boltzmann constant
        self.M_sun = 1.989e33           # Solar mass
        self.R_sun = 6.957e10           # Solar radius
        self.L_sun = 3.828e33           # Solar luminosity
        self.AU = 1.496e13              # Astronomical Unit
        self.mu = 2.34                  # Mean molecular weight

    def _load_observational_relations(self):
        """Load empirical relations from observations"""
        # From recent ALMA surveys and theoretical models
        self.empirical_relations = {
            'mdisk_mstar': {
                'slope': 1.0,       # log(M_disk) vs log(M_star) slope
                'scatter': 0.5,     # Natural logarithmic scatter
                'intercept': -1.5   # Typical scaling
            },
            'rout_mstar': {
                'slope': 38.0,      # AU per solar mass
                'scatter': 10.0     # AU
            },
            'temperature': {
                'q_range': (-0.6, -0.4),  # Radial temperature power law
                'T_1AU': (200, 300)       # Temperature at 1 AU range
            },
            'metallicity': {
                'mean': 1.0,        # Solar metallicity
                'scatter': 0.2      # Log-normal scatter
            },
            'B_field_density': {
                'slope': 0.5,       # B ∝ n_H^slope
                'scatter': 0.2      # Log-normal scatter
            }
        }

    def generate_single_ppd(self) -> PPDParameters:
        """
        Generate a single physically consistent PPD.
        Returns:
            PPDParameters: The generated PPD parameters.
        """
        params_dict = self.generate_parameters()
        
        # Compute Sigma0
        Sigma0 = self.compute_Sigma0(params_dict['disc_m'], params_dict['R_in'], params_dict['R_out'], params_dict['pindex'])
                
        # Create PPDParameters instance
        params = PPDParameters(
            m1=params_dict['m1'],
            accr1=params_dict['accr1'],
            J2_body1=params_dict['J2_body1'],
            disc_m=params_dict['disc_m'],
            Sigma0=Sigma0,
            R_in=params_dict['R_in'],
            R_ref=params_dict['R_ref'],
            R_out=params_dict['R_out'],
            H_R=params_dict['H_R'],
            pindex=params_dict['pindex'],
            qindex=params_dict['qindex'],
            dust_to_gas=params_dict['dust_to_gas'],
            grainsize=params_dict['grainsize'],
            graindens=params_dict['graindens'],
            beta_cool=params_dict['beta_cool'],
            planets=params_dict['planets']
        )
        return params

    def compute_Sigma0(self, disc_m: float, R_in: float, R_out: float, pindex: float) -> float:
        """Compute Sigma0 based on disc mass and radii"""
        r0 = 1.0  # Reference radius in AU
        if pindex == 1.0:
            Sigma0 = disc_m / (2 * np.pi * r0**2 * np.log(R_out / R_in))
        else:
            Sigma0 = (disc_m * (2 - pindex) / (2 * np.pi * r0**2) /
                      (R_out**(2 - pindex) - R_in**(2 - pindex)))
        return Sigma0

    def generate_parameters(self) -> Dict:
        """
        Generate complete set of physically consistent disk parameters.
        Returns dictionary of parameters in simulation units.
        """
        # Sample metallicity
        Z = self._sample_metallicity()

        # Sample cloud core properties
        core_mass = self._sample_core_mass()
        core_density = self._compute_core_density(core_mass)
        B_field = self._compute_magnetic_field(core_density)
        turbulence_alpha = self._sample_turbulence()
        rotational_beta = self._sample_rotation()
        mu = self._mass_to_flux_ratio(core_mass, B_field)

        # Stellar mass after collapse
        m_star = self._compute_stellar_mass(core_mass, mu)
        self._last_mstar = m_star  # Store for disk fraction calculation

        # Disk mass fraction influenced by magnetic braking and turbulence
        disk_fraction = self._compute_disk_fraction(rotational_beta, mu, turbulence_alpha)
        m_disk = disk_fraction * core_mass

        # Apply constraints
        m_star = np.clip(m_star, self.constraints.min_star_mass, self.constraints.max_star_mass)
        m_disk = np.clip(m_disk, self.constraints.min_disk_mass, self.constraints.max_disk_mass)

        # Compute disk radii
        R_in, R_ref, R_out = self._compute_disk_radii(m_star, m_disk, rotational_beta, mu)

        # Temperature profile considering external radiation
        qindex, h_r, T_ref = self._compute_temperature_profile(m_star, R_ref, Z)

        # Surface density profile
        pindex = self._compute_surface_density(m_disk, R_in, R_out)

        # Dust properties with metallicity effects
        dtg, grain_size, grain_dens = self._compute_dust_properties(Z)

        # Compute beta_cool
        beta_cool = self._compute_beta_cool(T_ref, m_star, m_disk, R_ref, h_r)

        # Accretion radius and oblateness
        accr1 = self._compute_accretion_radius(m_star)
        J2_body1 = 0.0

        # Generate planets
        planets = self.generate_planet_system(m_star, m_disk, R_in, R_out)

        # Compile parameters
        params = {
            'm1': m_star,
            'accr1': accr1,
            'J2_body1': J2_body1,
            'R_in': R_in,
            'R_ref': R_ref,
            'R_out': R_out,
            'disc_m': m_disk,
            'pindex': pindex,
            'qindex': qindex,
            'H_R': h_r,
            'dust_to_gas': dtg,
            'grainsize': grain_size,
            'graindens': grain_dens,
            'beta_cool': beta_cool,
            'planets': planets
        }

        # Optionally compute Q_out? Nonessential
        Q_out = self._compute_toomre_Q(params)
        params['Q_out'] = Q_out  # Include in output for reference

        return params

    def _sample_metallicity(self) -> float:
        """Sample metallicity relative to solar"""
        Z = self.rng.lognormal(mean=np.log(self.empirical_relations['metallicity']['mean']),
                               sigma=self.empirical_relations['metallicity']['scatter'])
        Z = np.clip(Z, self.constraints.min_metallicity, self.constraints.max_metallicity)
        return Z

    def _sample_core_mass(self) -> float:
        """
        Sample core mass using an IMF-inspired distribution.
        Returns mass in solar masses.
        """
        # Define mass ranges for different stellar types
        mass_ranges = {
            'brown_dwarf': (0.08, 0.15),    # Brown dwarf transition to very low mass
            'low_mass': (0.15, 0.5),        # M dwarfs
            'solar_type': (0.5, 2.0),       # K, G, and early F stars
            'intermediate': (2.0, 2.5)      # Late A stars/Herbig Ae
        }
        
        # Probability weights for each range (adjust these to match observed distributions)
        weights = {
            'brown_dwarf': 0.15,    # Lower probability for brown dwarfs
            'low_mass': 0.45,       # Highest probability for low mass stars
            'solar_type': 0.30,     # Moderate probability for solar-type
            'intermediate': 0.10     # Lower probability for massive stars
        }
        
        # First select which mass range we're sampling from
        range_choice = self.rng.choice(list(weights.keys()), p=list(weights.values()))
        min_mass, max_mass = mass_ranges[range_choice]
        
        print(f"\nMass sampling:")
        print(f"Selected mass range: {range_choice}")
        print(f"Mass range: {min_mass:.2f} - {max_mass:.2f} M_sun")
        
        # Sample mass using a combination of power-law and log-normal distribution
        if range_choice == 'brown_dwarf':
            # Use power law for brown dwarfs
            alpha = -0.3  # Brown dwarf mass function slope
            mass = ((max_mass**(alpha + 1) - min_mass**(alpha + 1)) * 
                   self.rng.random() + min_mass**(alpha + 1))**(1/(alpha + 1))
        
        elif range_choice in ['low_mass', 'solar_type']:
            # Log-normal distribution for low to solar mass stars
            log_min = np.log10(min_mass)
            log_max = np.log10(max_mass)
            mu = (log_min + log_max) / 2
            sigma = (log_max - log_min) / 4
            
            # Sample until we get a mass in the desired range
            while True:
                mass = 10**self.rng.normal(mu, sigma)
                if min_mass <= mass <= max_mass:
                    break
        
        else:  # intermediate mass
            # Salpeter-like IMF for more massive stars
            alpha = -2.35  # Salpeter IMF slope
            mass = ((max_mass**(alpha + 1) - min_mass**(alpha + 1)) * 
                   self.rng.random() + min_mass**(alpha + 1))**(1/(alpha + 1))
        
        print(f"Sampled initial mass: {mass:.3f} M_sun")
        return mass

    def _compute_core_density(self, core_mass: float) -> float:
        """Compute core density assuming a fixed radius"""
        core_radius = 0.1 * self.AU  # Fixed small radius in cm
        volume = (4/3) * np.pi * core_radius**3
        mass_cgs = core_mass * self.M_sun
        density = mass_cgs / volume  # g/cm^3
        return density

    def _compute_magnetic_field(self, density: float) -> float:
        """Compute magnetic field strength based on density"""
        rel = self.empirical_relations['B_field_density']
        log_B = rel['slope'] * np.log10(density) + self.rng.normal(0, rel['scatter'])
        B_field = 10**log_B
        B_field = np.clip(B_field, self.constraints.min_B_field, self.constraints.max_B_field)
        return B_field

    def _sample_turbulence(self) -> float:
        """Sample turbulent support parameter alpha"""
        alpha = 10**self.rng.normal(-2.0, 0.3)
        return alpha

    def _sample_rotation(self) -> float:
        """Sample rotational energy ratio beta"""
        # Typical values for molecular cloud cores are β ~ 0.02-0.1
        # Using lognormal distribution centered around β ~ 0.04
        beta = 10**self.rng.normal(-1.4, 0.3)  # This will give typical values around 0.02-0.1
        return beta

    def _mass_to_flux_ratio(self, core_mass: float, B_field: float) -> float:
        """Compute mass-to-flux ratio mu"""
        # Normalize to critical mass-to-flux ratio
        mu = (core_mass / B_field) / 5.0  # Divided by 5.0 to bring into reasonable range
        return np.clip(mu, 1.0, 10.0)  # Keep mu in physically reasonable range


    def _compute_stellar_mass(self, core_mass: float, mu: float) -> float:
        """
        Compute stellar mass with improved physics for pre-stellar cores.
        
        Args:
            core_mass: Initial core mass in solar masses
            mu: Mass-to-flux ratio
        
        Returns:
            Final stellar mass in solar masses
        """
        # Base mass loss fraction depends on core mass
        if core_mass < 0.5:
            # Higher efficiency for low mass cores
            base_loss = 0.2 + 0.1 * self.rng.random()
        elif core_mass < 2.0:
            # Moderate mass loss for solar-type stars
            base_loss = 0.3 + 0.15 * self.rng.random()
        else:
            # Higher mass loss for more massive stars
            base_loss = 0.4 + 0.2 * self.rng.random()
            
        # Magnetic effects on mass loss
        # Stronger fields (lower mu) increase mass loss
        mag_factor = 0.1 * np.exp(-mu/5)
        
        # Total mass loss fraction
        mass_loss_fraction = min(0.9, base_loss + mag_factor)
        
        # Calculate final stellar mass
        m_star = core_mass * (1 - mass_loss_fraction)
        
        # Ensure mass stays within physical limits
        m_star = np.clip(m_star, self.constraints.min_star_mass, self.constraints.max_star_mass)
        
        print(f"Mass loss calculation:")
        print(f"Initial core mass: {core_mass:.3f} M_sun")
        print(f"Base mass loss fraction: {base_loss:.3f}")
        print(f"Magnetic mass loss factor: {mag_factor:.3f}")
        print(f"Total mass loss fraction: {mass_loss_fraction:.3f}")
        print(f"Final stellar mass: {m_star:.3f} M_sun")
        
        return m_star
    
    def _compute_disk_fraction(self, beta: float, mu: float, alpha: float) -> float:
            """
            Compute disk mass fraction with improved mass dependence
            """
            # Base disk fraction from rotation
            f_base = 0.1 * (beta / 0.02)**0.5
            
            # Magnetic braking reduces disk mass
            f_mag = np.exp(-mu / 5)
            
            # Turbulence can enhance disk formation
            f_turb = (alpha / 0.01)**0.2
            
            # Scale disk fraction with stellar mass
            # Higher mass stars tend to have more massive disks
            m_star_factor = (self._last_mstar / 0.5)**0.5 if hasattr(self, '_last_mstar') else 1.0
            
            disk_fraction = f_base * f_mag * f_turb * m_star_factor
            disk_fraction = np.clip(disk_fraction, 0.01, 0.3)
            
            print(f"\nDisk fraction calculation:")
            print(f"Base fraction: {f_base:.3f}")
            print(f"Magnetic factor: {f_mag:.3f}")
            print(f"Turbulence factor: {f_turb:.3f}")
            print(f"Stellar mass factor: {m_star_factor:.3f}")
            print(f"Final disk fraction: {disk_fraction:.3f}")
            
            return disk_fraction

    def _compute_disk_radii(self, m_star: float, m_disk: float, beta: float, mu: float) -> Tuple[float, float, float]:
        """Compute disk inner, reference, and outer radii with refined physics."""
        # Inner radius using dust sublimation temperature with variable composition
        T_sub = 1500  # K, average dust sublimation temperature
        L_star = self._compute_stellar_luminosity(m_star)
        R_in = np.sqrt(L_star / (16 * np.pi * self.sigma_sb * T_sub**4)) / self.AU
        R_in = max(R_in, self.constraints.min_r_in)  # Enforce constraints

        # Outer radius based on angular momentum and disk mass scaling
        R_c = 2000 * beta * (m_star / 0.08)**0.5  # Rotational scaling
        R_mass = 200 * (m_disk / 1e-4)**0.5  # Disk mass scaling
        R_out = max(R_c, R_mass)

        # Magnetic braking adjustment for outer radius
        mag_factor = 0.5 * (1.0 + np.tanh((mu - 5.0) / 2.0))  # Smooth transition around mu=5
        R_out *= mag_factor

        # Add scatter to account for observational variability
        scatter = self.rng.normal(1.0, 0.2)
        R_out *= scatter

        # Apply constraints to the outer radius
        R_out = np.clip(R_out, max(50.0, 2.0 * R_in), self.constraints.max_r_out)

        # Reference radius is geometric mean of inner and outer radii
        R_ref = np.sqrt(R_in * R_out)
        return R_in, R_ref, R_out

    def _compute_stellar_luminosity(self, m_star: float) -> float:
        """Compute pre-main sequence stellar luminosity using Baraffe models."""
        # Based on Baraffe et al. (2015) models for 1 Myr old stars
        if m_star < 0.1:
            L_star = self.L_sun * (m_star / 0.1)**1.7  # Steeper relation for very low mass stars
        else:
            L_star = self.L_sun * (m_star)**3.5  # Standard PMS relation
        return L_star

    def _compute_temperature_profile(self, m_star: float, r_ref: float, Z: float) -> Tuple[float, float, float]:
        """Compute temperature power law parameters with refined metallicity effects."""
        # Radial temperature profile
        temp_range = self.empirical_relations['temperature']
        qindex = - self.rng.uniform(*temp_range['q_range'])

        # Include external radiation and stellar luminosity
        T_ext = 10  # K, background radiation temperature
        L_star = self._compute_stellar_luminosity(m_star)
        T_star = (L_star / (4 * np.pi * (r_ref * self.AU)**2 * self.sigma_sb))**0.25
        T_ref = (T_star**4 + T_ext**4)**0.25

        # Adjust temperature for metallicity
        T_ref *= Z**0.05  # Slight effect on opacity

        # Aspect ratio (H/R)
        cs = np.sqrt(self.k_B * T_ref / (self.mu * self.m_p))
        v_k = np.sqrt(self.G * m_star * self.M_sun / (r_ref * self.AU))
        H_R = cs / v_k
        H_R *= (1 + self._sample_turbulence())  # Turbulent pressure support
        H_R = np.clip(H_R, self.constraints.min_aspect, self.constraints.max_aspect)
        return qindex, H_R, T_ref


    def _compute_surface_density(self, m_disk: float, r_in: float, r_out: float) -> float:
        """Compute surface density power law index with observational constraints."""
        # Empirical studies suggest surface density indices between 1.0 and 0.5
        pindex = self.rng.uniform(1.0, 0.5)
        return pindex


    def _compute_dust_properties(self, Z: float) -> Tuple[float, float, float]:
        """Compute dust properties with metallicity effects"""
        # Dust-to-gas ratio scales with metallicity
        base_dtg = 0.01  # ISM value
        dtg = base_dtg * Z
        dtg = np.clip(dtg, self.constraints.min_dtg, self.constraints.max_dtg)

        # Grain size distribution with metallicity influence
        a_min = 1e-5  # cm
        a_max = 0.1 * Z  # Larger grains with higher metallicity
        a_max = np.clip(a_max, self.constraints.min_grain, self.constraints.max_grain)
        q = -3.5  # MRN slope

        # Compute characteristic grain size from distribution
        grainsize = np.exp(np.mean(np.log([a_min, a_max])))

        # Grain density
        grain_dens = 3.0  # g/cm^3, standard value
        return dtg, grainsize, grain_dens

    def _compute_accretion_radius(self, m_star: float) -> float:
        """Compute accretion radius for the central star"""
        R_star = self._compute_stellar_radius(m_star)
        accr_radius = 5 * R_star / self.AU  # In AU
        return accr_radius

    def _compute_stellar_radius(self, m_star: float) -> float:
        """Estimate stellar radius for pre-main-sequence stars"""
        R_star = self.R_sun * (m_star / 1.0)**0.8
        return R_star

    def _compute_beta_cool(self, T_ref: float, m_star: float, m_disk: float, r_ref: float, H_R: float) -> float:
        """Compute beta_cool parameter with refined opacity and optical depth."""
        # Midplane temperature considering viscous and radiative contributions
        alpha_visc = 0.01  # Viscous alpha
        T_visc = (3 * self.G * m_star * self.M_sun * m_disk * self.M_sun * alpha_visc) / (8 * np.pi * self.sigma_sb * (r_ref * self.AU)**3)
        T_mid = (T_ref**4 + T_visc)**0.25

        # Compute sound speed
        cs = np.sqrt(self.k_B * T_mid / (self.mu * self.m_p))

        # Surface density from mass and radii
        Sigma = self.compute_Sigma0(m_disk, r_ref, r_ref * 100, -1)

        # Opacity laws (simplified Bell & Lin 1994)
        if T_mid < 160:
            kappa = 2e-4 * T_mid**2  # Ice grains
        elif T_mid < 1500:
            kappa = 2e16 * T_mid**(-7)  # Sublimation of ice
        else:
            kappa = 1e1  # Gas phase

        tau = kappa * Sigma  # Optical depth

        # Cooling time
        if tau > 1:
            t_cool = (3 * kappa * Sigma * cs**2) / (16 * self.sigma_sb * T_mid**4)
        else:
            t_cool = (3 * cs**2) / (16 * self.sigma_sb * T_mid**4 * kappa * Sigma)

        # Orbital frequency
        Omega = np.sqrt(self.G * m_star * self.M_sun / (r_ref * self.AU)**3)

        # Compute beta_cool
        beta_cool = t_cool * Omega
        beta_cool = np.clip(beta_cool, 1.0, 50.0)  # Ensure valid range
        print(f"Beta cool parameter: {beta_cool}")
        return beta_cool

    def _compute_toomre_Q(self, params: Dict) -> float:
        """Compute Toomre Q parameter with refined surface density and sound speed."""
        h_out = params['H_R'] * params['R_out'] * (params['R_out'] / params['R_ref'])**(params['qindex'] + 0.5)
        sigma_out = params['disc_m'] * self.M_sun / (2 * np.pi * (params['R_out'] * self.AU)**2)
        omega_out = np.sqrt(self.G * params['m1'] * self.M_sun / (params['R_out'] * self.AU)**3)
        cs_out = h_out * omega_out
        Q_out = cs_out * omega_out / (np.pi * self.G * sigma_out)
        return Q_out

    def generate_planet_system(self, stellar_mass: float, disk_mass: float,
                               R_in: float, R_out: float) -> List[PlanetParameters]:
        # Generate physically consistent planetary system
        # Determine number of planets with limits
        max_planets = min(6, int(disk_mass / 0.005))
        n_planets = np.random.randint(0, max_planets + 1)

        if n_planets == 0:
            return []

        # Set a margin to avoid planets lying exactly on R_in or R_out
        margin = 0.1 * (R_out - R_in)

        # Generate planet locations within disk boundaries
        available_radii = np.linspace(R_in + margin, R_out - margin, 1000)
        planet_radii = np.sort(np.random.choice(available_radii, n_planets, replace=False))

        planets = []
        for radius in planet_radii:
            max_mass = min(
                10.0,  # Maximum planet mass (M_Jupiter)
                (disk_mass / 0.1) * (radius / 30)**(-1.5)  # Mass decreases with radius
            )
            mass = np.random.uniform(0.1, max_mass) # chooses randomly from uniform distribution, low = 0.1 and high = max_mass

            # Inclination dependent on system mass
            max_incl = 5 * (stellar_mass / disk_mass)**0.2  # reduced this from original val -- can be computed from orbital momentum but eh
            incl = np.random.rayleigh(max_incl / 3)
            incl = min(incl, 15)  # inclination limit

            # Sample accretion radius and J2 moment
            accr_radius = radius * (mass / (3*stellar_mass)) ** (1/3) # R_H = a (M_p / 3M_s)^(1/3)

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

    def generate_parameter_set(self, n_discs: int) -> List[PPDParameters]:
        """
        Generate parameters for multiple discs.
        Parameters:
            n_discs (int): Number of discs to generate.
        Returns:
            List of PPDParameters.
        """
        return [self.generate_single_ppd() for _ in range(n_discs)]
