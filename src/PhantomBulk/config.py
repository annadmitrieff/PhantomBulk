#!/usr/bin/env python3
# src/config.py

from pathlib import Path
import yaml
import os
import logging

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Assign attributes with proper type casting and path expansion
        self.VENV_PATH = Path(os.path.expandvars(cfg.get('VENV_PATH', ''))).expanduser()
        self.PHANTOM_DIR = Path(os.path.expandvars(cfg.get('PHANTOM_DIR', ''))).expanduser()
        self.PYTHON_SCRIPT = Path(os.path.expandvars(cfg.get('PYTHON_SCRIPT', ''))).expanduser()
        self.SETUP_TEMPLATE = Path(os.path.expandvars(cfg.get('SETUP_TEMPLATE', ''))).expanduser()
        self.REFERENCE_FILE = Path(os.path.expandvars(cfg.get('REFERENCE_FILE', ''))).expanduser()
        self.PHANTOM_SETUP_TYPE = cfg.get('PHANTOM_SETUP_TYPE', 'default')
        self.JOB_SCHEDULER = cfg.get('JOB_SCHEDULER', 'SLURM')
        
        # Explicitly cast to integers where appropriate
        self.CPUS_PER_TASK = int(cfg.get('CPUS_PER_TASK', 20))
        self.PARTITION = cfg.get('PARTITION', 'batch')
        self.MAIL_TYPE = cfg.get('MAIL_TYPE', '')
        self.N_TASKS = int(cfg.get('N_TASKS', 1))
        self.TIME = str(cfg.get('TIME', '6-23:59:59'))
        self.MEM = str(cfg.get('MEM', '10G'))
        self.USER_EMAIL = cfg.get('USER_EMAIL', '')
        self.N_SIMS = int(cfg.get('N_SIMS', 10))
        self.OUTPUT_DIR = Path(os.path.expandvars(cfg.get('OUTPUT_DIR', '$HOME/PhantomBulk/outputs/'))).expanduser()
        self.log_level = cfg.get('log_level', 'INFO')
        self.seed = int(cfg.get('seed', 42))
        self.parameter_ranges = cfg.get('parameter_ranges', {})
        self.MCFOST_EXEC = Path(os.path.expandvars(cfg.get('MCFOST_EXEC', ''))).expanduser()
        self.LD_LINUX = Path(os.path.expandvars(cfg.get('LD_LINUX', ''))).expanduser()

        # Convert all parameter ranges from strings to floats
        for param, ranges in self.parameter_ranges.items():
            for range_type in ['core', 'tail']:
                # Check if elements are strings and convert to floats
                if isinstance(ranges[range_type][0], str):
                    self.parameter_ranges[param][range_type] = [float(x) for x in ranges[range_type]]

        # Logging for debugging purposes
        logging.debug(f"CPUS_PER_TASK: {self.CPUS_PER_TASK} (type: {type(self.CPUS_PER_TASK)})")
        logging.debug(f"N_TASKS: {self.N_TASKS} (type: {type(self.N_TASKS)})")
        logging.debug(f"N_SIMS: {self.N_SIMS} (type: {type(self.N_SIMS)})")
        logging.debug(f"seed: {self.seed} (type: {type(self.seed)})")
        
        # Validate paths
        for attr in ['VENV_PATH', 'PHANTOM_DIR', 'PYTHON_SCRIPT', 'SETUP_TEMPLATE', 'REFERENCE_FILE', 'MCFOST_EXEC', 'LD_LINUX']:
            path = getattr(self, attr)
            if not path.exists():
                logging.error(f"The path for '{attr}' does not exist: {path}")
                raise FileNotFoundError(f"The path for '{attr}' does not exist: {path}")
            else:
                logging.debug(f"Verified '{attr}' exists at: {path}")
