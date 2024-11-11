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
        if setup_type == "default":
            setup_path = config.SETUP_TEMPLATE
        else:
            setup_path = config.SETUP_TEMPLATE.parent / f"{setup_type}.setup"
            if not setup_path.exists():
                logging.warning(f"Setup template for '{setup_type}' not found. Using default setup.")
                setup_path = config.SETUP_TEMPLATE

        if setup_path.exists():
            logging.info(f"Using setup template '{setup_path}'.")
            return setup_path
        else:
            logging.error(f"Setup template '{setup_path}' does not exist.")
            raise FileNotFoundError(f"Setup template '{setup_path}' does not exist.")

    def read_file(self, filename: Path) -> str:
        """Read the content of a file."""
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
            for exe in executables:
                exe_path = custom_dir_path / exe
                if exe_path.is_file() and os.access(exe_path, os.X_OK):
                    executable_paths.append(exe_path)
                    logging.info(f"Found executable '{exe}' at '{exe_path}'.")
                else:
                    logging.warning(f"Executable '{exe}' not found or not executable in '{custom_dir_path}'.")

        # Search in system PATH for any missing executables
        for exe in executables:
            if not any(path.name == exe for path in executable_paths):
                path = shutil.which(exe)
                if path:
                    executable_paths.append(Path(path))
                    logging.info(f"Found executable '{exe}' in system PATH at '{path}'.")
                else:
                    logging.error(f"Executable '{exe}' not found in system PATH.")

        if len(executable_paths) != len(executables):
            raise FileNotFoundError("One or more PHANTOM executables not found.")

        return tuple(executable_paths)  # (phantom_path, phantomsetup_path)
