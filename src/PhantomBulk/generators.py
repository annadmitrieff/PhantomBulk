# src/PhantomBulk/generators.py

import numpy as np
import logging
import traceback
from typing import List, Tuple
from .utils import sample_parameter
from .config import Config
from .data_classes import PPDParameters, PlanetParameters

class PhysicalPPDGenerator:
    """Generate PPD parameters with physical correlations."""

    def __init__(self, config: Config):
        """Initialize the generator with configuration parameters."""
        self.config = config
        np.random.seed(self.config.seed)
        self.load_survey_distributions()

    def load_survey_distributions(self):
        """Load  distributions from config."""
        self.parameter_ranges = self.config.parameter_ranges

    def compute_temperature_structure(self, stellar_mass: float) -> Tuple[float, float]:
        """
        Compute disc temperature structure.

        Parameters:
            stellar_mass (float): Stellar mass in solar masses.

        Returns:
            Tuple containing T0 (Kelvin) and qindex (power-law index).
        """
        # Temperature scaling with stellar luminosity
        L_star = stellar_mass ** 3.5  # Approximate luminosity scaling: L \propto M_star ** 3.5
        T0 = 280 * (L_star ** 0.25)  # Temperature at 1 AU: For T_sun, T0 = 280K
        #qindex = np.random.normal(0.25, 0.75)
        qindex = 0.5
        return T0, qindex

    def compute_disc_structure(
        self, stellar_mass: float, T0: float, qindex: float
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute physically consistent disc structure.

        Parameters:
            stellar_mass (float): Stellar mass in solar masses.
            T0 (float): Temperature at reference radius.
            qindex (float): Temperature power-law index.

        Returns:
            Tuple containing disc_mass, R_out, R_in, Sigma0, pindex.
        """
        # Disc mass as a fraction of stellar mass
        disc_mass = 0.01 * stellar_mass

        # Outer radius sampled from parameter ranges
        R_out = sample_parameter(
            self.parameter_ranges['R_out']['core'],
            self.parameter_ranges['R_out']['tail']
        )
        # Inner radius sampled from parameter ranges
        R_in = sample_parameter(
            self.parameter_ranges['R_in']['core'],
            self.parameter_ranges['R_in']['tail']
        )
        if R_in >= R_out:
            raise ValueError(f"Invalid radii: R_in ({R_in}) must be less than R_out ({R_out}).")

        # Sample p-index from a distribution
        # pindex = np.random.uniform(0.5, 1.5)
        pindex = 1.0

        # Reference radius for normalization
        r0 = np.sqrt(R_in, R_out)

        # Calculate Sigma0 based on disc_mass, R_in, R_out, and pindex
        if pindex == 1.0:
            # Special case where pindex equals 1
            Sigma0 = disc_mass / (2 * np.pi * r0**2 * np.log(R_out / R_in)) # = disc mass / (area of circle with rad as geom mean of R_in and R_out * ln(R_out / R_in))
        # else:
            # General case (removed because pindex = 1)
            # Sigma0 = (disc_mass * (2 - pindex) / (2 * np.pi * r0**2) /
                    # (R_out**(2 - pindex) - R_in**(2 - pindex)))

        return disc_mass, R_out, R_in, Sigma0, pindex


    def compute_reference_radius(self, R_in: float, R_out: float, Sigma0: float, pindex: float) -> float:
        """
        Compute a reference radius (R_ref) for the disc.

        Parameters:
            R_in (float): Inner radius in AU.
            R_out (float): Outer radius in AU.
            Sigma0 (float): Surface density normalization in g/cm^2.
            pindex (float): Surface density power-law index.

        Returns:
            float: Computed or selected R_ref.

        """
        if pindex != 1.0:
            R_ref = ((Sigma0 * (2 - pindex)) / (np.pi * (R_out**(2 - pindex) - R_in**(2 - pindex)))) ** (1 / pindex)
        else:
            R_ref = np.sqrt(R_in * R_out) # Ref radius as geometric mean; same as for sigma0 computation

        # Ensure R_ref lies within valid bounds
        R_ref = max(R_in, min(R_ref, R_out))
        return R_ref

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
        k_B = 1.380649e-16  # Boltzmann constant in erg/K
        mu = 2.34           # Mean molecular weight for molecular H2
        m_H = 1.6735575e-24 # Hydrogen mass in g
        c_s = np.sqrt(k_B * T0 / (mu * m_H)) # soundspeed; contingent on gas temp + mean mol weight + hydrogen mass (ideal gas law)
        Omega = np.sqrt(1.0 / (R_ref * 1.496e13)**3 * (6.67430e-8 * stellar_mass * 1.98847e33))  # rad/s
        H = c_s / Omega  # cm; scale height
        H_AU = H / 1.496e13
        H_R = H_AU / R_ref
        return H_R # height to radius ratio

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
            accr_radius = radius (mass / (3*stellar_mass)) ** (1/3) # R_H = a (M_p / 3M_s)^(1/3)

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
    def validate_parameters(self, params: PPDParameters) -> bool:
        """
        Validate generated PPD parameters.
        Parameters:
            params (PPDParameters): The generated PPD parameters.
        Returns:
            bool: True if valid, False otherwise.
        """
        if params.R_in >= params.R_out:
            return False
        if params.disc_m <= 0:
            return False
        if params.H_R <= 0 or params.H_R > 0.5:
            return False
        for planet in params.planets:
            if planet.radius <= params.R_in or planet.radius >= params.R_out:
                return False
            if planet.mass <= 0:
                return False
        return True

    def generate_single_ppd(self) -> PPDParameters:
        """
        Generate a single physically consistent PPD.

        Returns:
            PPDParameters: The generated PPD parameters.

        Raises:
            ValueError: If invalid parameters are generated after multiple attempts.
        """
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                # Sample stellar mass
                stellar_mass = sample_parameter(
                    self.parameter_ranges['m1']['core'],
                    self.parameter_ranges['m1']['tail']
                )
                logging.debug(f"Attempt {attempt}: Generated stellar_mass = {stellar_mass} (type: {type(stellar_mass)})")

                # Compute temperature structure
                T0, qindex = self.compute_temperature_structure(stellar_mass)
                logging.debug(f"Attempt {attempt}: Computed temperature structure: T0={T0}, qindex={qindex} (types: T0={type(T0)}, qindex={type(qindex)})")

                # Compute disc structure
                disc_mass, R_out, R_in, Sigma0, pindex = self.compute_disc_structure(stellar_mass, T0, qindex)
                logging.debug(f"Attempt {attempt}: Computed disc structure: disc_mass={disc_mass}, R_out={R_out}, R_in={R_in}, Sigma0={Sigma0}, pindex={pindex} (types: disc_mass={type(disc_mass)}, R_out={type(R_out)}, R_in={type(R_in)}, Sigma0={type(Sigma0)}, pindex={type(pindex)})")

                # Compute reference radius
                R_ref = self.compute_reference_radius(R_in, R_out, Sigma0, pindex)
                logging.debug(f"Attempt {attempt}: Computed R_ref = {R_ref}")

                # Compute aspect ratio
                H_R = self.compute_aspect_ratio(T0, stellar_mass)
                logging.debug(f"Attempt {attempt}: Computed aspect ratio H_R={H_R} (type: {type(H_R)})")

                # Sample additional parameters
                accr1 = float(R_in - 0.01)
                logging.debug(f"Attempt {attempt}: Sampled accr1 = {accr1} (type: {type(accr1)})")

                J2_body1 = sample_parameter(
                    self.parameter_ranges['J2_body1']['core'],
                    self.parameter_ranges['J2_body1']['tail']
                )
                logging.debug(f"Attempt {attempt}: Sampled J2_body1 = {J2_body1} (type: {type(J2_body1)})")

                dust_to_gas = sample_parameter(
                    self.parameter_ranges['dust_to_gas']['core'],
                    self.parameter_ranges['dust_to_gas']['tail']
                )
                logging.debug(f"Attempt {attempt}: Sampled dust_to_gas = {dust_to_gas} (type: {type(dust_to_gas)})")

                grainsize = sample_parameter(
                    self.parameter_ranges['grainsize']['core'],
                    self.parameter_ranges['grainsize']['tail']
                )
                logging.debug(f"Attempt {attempt}: Sampled grainsize = {grainsize} (type: {type(grainsize)})")

                graindens = sample_parameter(
                    self.parameter_ranges['graindens']['core'],
                    self.parameter_ranges['graindens']['tail']
                )
                logging.debug(f"Attempt {attempt}: Sampled graindens = {graindens} (type: {type(graindens)})")

                beta_cool = sample_parameter(
                    self.parameter_ranges['beta_cool']['core'],
                    self.parameter_ranges['beta_cool']['tail']
                )
                logging.debug(f"Attempt {attempt}: Sampled beta_cool = {beta_cool} (type: {type(beta_cool)})")

                # Generate planetary system
                planets = self.generate_planet_system(stellar_mass, disc_mass, R_in, R_out)
                logging.debug(f"Attempt {attempt}: Generated {len(planets)} planets (type: {type(planets)})")

                # Create PPDParameters instance with all required fields
                params = PPDParameters(
                    m1=stellar_mass,
                    accr1=accr1,
                    J2_body1=J2_body1,
                    disc_m=disc_mass,
                    Sigma0=Sigma0,
                    R_in=R_in,
                    R_ref=R_ref,
                    R_out=R_out,
                    H_R=H_R,
                    pindex=pindex,
                    qindex=qindex,
                    dust_to_gas=dust_to_gas,
                    grainsize=grainsize,
                    graindens=graindens,
                    beta_cool=beta_cool,
                    T0=T0,
                    planets=planets
                )
                logging.debug(f"Attempt {attempt}: Created PPDParameters: {params}")

                # **Add Assertions to Ensure Correct Types**
                assert isinstance(params.m1, float), f"m1 must be float, got {type(params.m1)}"
                assert isinstance(params.accr1, float), f"accr1 must be float, got {type(params.accr1)}"
                assert isinstance(params.J2_body1, float), f"J2_body1 must be float, got {type(params.J2_body1)}"
                assert isinstance(params.disc_m, float), f"disc_m must be float, got {type(params.disc_m)}"
                assert isinstance(params.Sigma0, float), f"Sigma0 must be float, got {type(params.Sigma0)}"
                assert isinstance(params.R_in, float), f"R_in must be float, got {type(params.R_in)}"
                assert isinstance(params.R_ref, float), f"R_ref must be float, got {type(params.R_out)}"
                assert isinstance(params.R_out, float), f"R_out must be float, got {type(params.R_out)}"
                assert isinstance(params.H_R, float), f"H_R must be float, got {type(params.H_R)}"
                assert isinstance(params.pindex, float), f"pindex must be float, got {type(params.pindex)}"
                assert isinstance(params.qindex, float), f"qindex must be float, got {type(params.qindex)}"
                assert isinstance(params.dust_to_gas, float), f"dust_to_gas must be float, got {type(params.dust_to_gas)}"
                assert isinstance(params.grainsize, float), f"grainsize must be float, got {type(params.grainsize)}"
                assert isinstance(params.graindens, float), f"graindens must be float, got {type(params.graindens)}"
                assert isinstance(params.beta_cool, float), f"beta_cool must be float, got {type(params.beta_cool)}"
                assert isinstance(params.T0, float), f"T0 must be float, got {type(params.T0)}"
                assert isinstance(params.planets, list), f"planets must be a list, got {type(params.planets)}"

                # Validate parameters
                if self.validate_parameters(params):
                    logging.debug(f"Attempt {attempt}: Parameters validated successfully.")
                    return params
                else:
                    logging.warning(f"Attempt {attempt}: Parameters failed validation. Regenerating.")
            except AssertionError as e:
                logging.error(f"Attempt {attempt}: Assertion error during PPD generation: {e}")
                continue
            except Exception as e:
                logging.error(f"Attempt {attempt}: Error generating PPD: {e}")
                logging.error(traceback.format_exc())
                continue

        raise ValueError("Failed to generate a valid PPD after multiple attempts.")

    def generate_parameter_set(self, n_discs: int) -> List[PPDParameters]:
        """
        Generate parameters for multiple discs.

        Parameters:
            n_discs (int): Number of discs to generate.

        Returns:
            List of PPDParameters.
        """
        return [self.generate_single_ppd() for _ in range(n_discs)]
