#!/usr/bin/env python3
# src/utils.py

import numpy as np

def sample_parameter(core_range: tuple, tail_range: tuple, tail_probability: float = 0.2) -> float:
    """
    Sample a parameter with a probability to select from the tail range.

    Args:
        core_range (tuple): The (min, max) for the core range.
        tail_range (tuple): The (min, max) for the tail range.
        tail_probability (float): Probability of selecting from the tail range.

    Returns:
        float: Sampled parameter value.
    """
    if np.random.random() < tail_probability:
        return np.random.uniform(*tail_range)
    else:
        return np.random.uniform(*core_range)
