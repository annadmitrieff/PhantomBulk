#!/usr/bin/env python3
# src/config.py

from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict


class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Assign attributes
        self.VENV_PATH = Path(cfg.get('VENV_PATH', ''))
        self.PHANTOM_DIR = Path(cfg.get('PHANTOM_DIR', ''))
        self.PYTHON_SCRIPT = Path(cfg.get('PYTHON_SCRIPT', ''))
        self.SETUP_TEMPLATE = Path(cfg.get('SETUP_TEMPLATE', ''))
        self.REFERENCE_FILE = Path(cfg.get('REFERENCE_FILE', ''))
        self.PHANTOM_SETUP_TYPE = cfg.get('PHANTOM_SETUP_TYPE', 'default')
        self.JOB_SCHEDULER = cfg.get('JOB_SCHEDULER', 'SLURM')
        self.CPUS_PER_TASK = cfg.get('CPUS_PER_TASK', '20')
        self.PARTITION = cfg.get('PARTITION', 'batch')
        self.MAIL_TYPE = cfg.get('MAIL_TYPE', '')
        self.N_TASKS = cfg.get('N_TASKS', '1')
        self.TIME = cfg.get('TIME', '6-23:59:59')
        self.MEM = cfg.get('MEM', '10G')
        self.USER_EMAIL = cfg.get('USER_EMAIL', '')
        self.N_SIMS = cfg.get('N_SIMS', 10)
        self.OUTPUT_DIR = Path(cfg.get('OUTPUT_DIR', '$HOME/PhantomBulk/outputs/')).expanduser()
        self.log_level = cfg.get('log_level', 'INFO')
        self.seed = cfg.get('seed', 42)
        self.parameter_ranges = cfg.get('parameter_ranges', {})
        self.MCFOST_EXEC = Path(cfg.get('MCFOST_EXEC', ''))
        self.LD_LINUX = Path(cfg.get('LD_LINUX', ''))

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
