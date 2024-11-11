#!/usr/bin/env python3
# src/submitter.py

import subprocess
import logging
from pathlib import Path
from .config import Config

class Submitter:
    """Handles job submission to cluster schedulers."""

    def __init__(self, config: Config):
        self.scheduler = config.JOB_SCHEDULER.upper()
        self.submit_cmd = self.get_submit_command()

    def get_submit_command(self) -> list:
        """Return the submission command based on the scheduler."""
        job_scheduler_map = {
            'SLURM': ['sbatch'],
            'PBS': ['qsub'],
            'SGE': ['qsub', '-cwd']
        }
        if self.scheduler not in job_scheduler_map:
            logging.error(f"Unsupported job scheduler: {self.scheduler}")
            raise ValueError(f"Unsupported job scheduler: {self.scheduler}")
        return job_scheduler_map[self.scheduler]

    def submit_job(self, job_script: Path):
        """Submit a single job script."""
        try:
            result = subprocess.run(self.submit_cmd + [str(job_script)], check=True)
            logging.info(f"Submitted job: {job_script}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit job {job_script}: {e}")
