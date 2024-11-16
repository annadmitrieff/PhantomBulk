#!/usr/bin/env python3
# src/PhantomBulk/file_manager.py

import os
import shutil
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from .config import Config
from .data_classes import PPDParameters, PlanetParameters  # Import data classes
import re  # Needed for regex operations

class PHANTOMFileManager:
    def __init__(self, config):
        self.setup_template = self.get_setup_template(config)
        self.phantom_executables = self.find_phantom_executables(config.PHANTOM_DIR)

    def get_setup_template(self, config) -> Path:
        """Select setup template based on PHANTOM_SETUP_TYPE."""
        setup_type = config.PHANTOM_SETUP_TYPE
        logging.debug(f"PHANTOM_SETUP_TYPE: {setup_type}")
        
        if setup_type == "default":
            setup_path = config.SETUP_TEMPLATE
            logging.debug(f"Using default setup template path: {setup_path}")
        else:
            setup_path = config.SETUP_TEMPLATE.parent / f"{setup_type}.setup"
            logging.debug(f"Looking for setup template at: {setup_path}")
            if not setup_path.exists():
                logging.warning(f"Setup template for '{setup_type}' not found. Using default setup.")
                setup_path = config.SETUP_TEMPLATE
                logging.debug(f"Reverting to default setup template path: {setup_path}")

        if setup_path.exists():
            logging.info(f"Using setup template '{setup_path}'.")
            return setup_path
        else:
            logging.error(f"Setup template '{setup_path}' does not exist.")
            raise FileNotFoundError(f"Setup template '{setup_path}' does not exist.")

    def read_file(self, filename: Path) -> str:
        """Read the content of a file."""
        logging.debug(f"Reading file: {filename}")
        if filename.is_file():
            with open(filename, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Template file '{filename}' not found.")

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
            logging.debug(f"Searching for executables in custom directory: {custom_dir_path}")
            for exe in executables:
                exe_path = custom_dir_path / exe
                logging.debug(f"Checking for executable: {exe_path}")
                if exe_path.is_file() and os.access(exe_path, os.X_OK):
                    executable_paths.append(exe_path)
                    logging.info(f"Found executable '{exe}' at '{exe_path}'.")
                else:
                    logging.warning(f"Executable '{exe}' not found or not executable in '{custom_dir_path}'.")

        # Search in system PATH for any missing executables
        for exe in executables:
            if not any(path.name == exe for path in executable_paths):
                path = shutil.which(exe)
                logging.debug(f"Searching for executable '{exe}' in system PATH.")
                if path:
                    executable_paths.append(Path(path))
                    logging.info(f"Found executable '{exe}' in system PATH at '{path}'.")
                else:
                    logging.error(f"Executable '{exe}' not found in system PATH.")

        if len(executable_paths) != len(executables):
            logging.error("One or more PHANTOM executables not found.")
            raise FileNotFoundError("One or more PHANTOM executables not found.")

        return tuple(executable_paths)  # (phantom_path, phantomsetup_path)

    def create_setup_file(self, params: PPDParameters, output_dir: Path, sim_id: int):
        """Generate the `.setup` file with all placeholders replaced by params."""
        # Step 1: Read the setup template content as a string
        try:
            setup_content = self.read_file(self.setup_template)
            logging.debug(f"Read setup template from '{self.setup_template}'.")
        except Exception as e:
            logging.error(f"Failed to read setup template: {e}")
            raise

        # Step 2: Number of planets
        num_planets = len(params.planets)

        # Step 3: Generate planet configurations
        planet_configurations = self.generate_planet_configurations(params.planets)

        # Step 4: Create dictionary for all parameter placeholders
        param_dict = {
            "m1": params.m1,
            "accr1": params.accr1,
            "J2_body1": params.J2_body1,
            "R_in": params.R_in,
            "R_ref": params.R_ref,
            "R_out": params.R_out,
            "disc_m": params.disc_m,
            "pindex": params.pindex,
            "qindex": params.qindex,
            "H_R": params.H_R,
            "dust_to_gas": params.dust_to_gas,
            "grainsize": params.grainsize,
            "graindens": params.graindens,
            "NUM_PLANETS": num_planets,
            "PLANET_CONFIGURATIONS": planet_configurations,
            "beta_cool": params.beta_cool,
        }

        # Step 5: Replace each placeholder with actual parameter values
        for key, value in param_dict.items():
            placeholder = f"{{{{{key}}}}}"  # Matches {{PARAM_NAME}} in templates
            if placeholder in setup_content:
                setup_content = setup_content.replace(placeholder, str(value))
                logging.debug(f"Replaced placeholder '{placeholder}' with '{value}'.")
            else:
                logging.warning(f"Placeholder '{placeholder}' not found in setup template.")

        # Step 6: Write the fully populated .setup file
        setup_file = output_dir / f'dustysgdisc.setup'  # LATER: IMPLEMENT DIFFERENT PREBAKED SETUPS
        try:
            with open(setup_file, 'w') as f:
                f.write(setup_content)
            logging.info(f"Generated setup file '{setup_file}' for simulation {sim_id}.")
        except Exception as e:
            logging.error(f"Failed to write setup file '{setup_file}': {e}")
            raise

    
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
            # r'^\s*beta_cool\s*=\s*\d+\.\d+\s*!.*$': f"      beta_cool = {params.beta_cool:.3f}    ! beta factor in Gammie (2001) cooling" #,
            # r'^\s*Tfloor\s*=\s*\d+\.\d+\s*!.*$': f"     Tfloor = {params.T0:.3f}    ! Temperature at 1 AU"
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