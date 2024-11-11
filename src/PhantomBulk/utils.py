#!/usr/bin/env python3
# src/utils.py

import logging
import numpy as np

def sample_parameter(core_range: list, tail_range: list, tail_probability: float = 0.2) -> float:
    """
    Sample a parameter with a probability to select from the tail range.

    Args:
        core_range (list): The [min, max] for the core range.
        tail_range (list): The [min, max] for the tail range.
        tail_probability (float): Probability of selecting from the tail range.

    Returns:
        float: Sampled parameter value.
    """
    # Ensure all range elements are floats
    try:
        core_min, core_max = float(core_range[0]), float(core_range[1])
        tail_min, tail_max = float(tail_range[0]), float(tail_range[1])
    except ValueError as e:
        logging.error(f"Non-numeric value in parameter ranges: {e}")
        raise

    if np.random.random() < tail_probability:
        sampled_value = float(np.random.uniform(tail_min, tail_max))
        logging.debug(f"Sampling from tail_range: [{tail_min}, {tail_max}] -> {sampled_value}")
        return sampled_value
    else:
        sampled_value = float(np.random.uniform(core_min, core_max))
        logging.debug(f"Sampling from core_range: [{core_min}, {core_max}] -> {sampled_value}")
        return sampled_value