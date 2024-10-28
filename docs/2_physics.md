Yes, I can help adjust the LaTeX in your document to be GitHub Markdown compliant. The main changes involve:

1. **Removing the outer triple backticks**: This ensures that your Markdown and LaTeX are processed correctly, as content within triple backticks is treated as code and not rendered.

2. **Adjusting headers and anchor links**: To ensure that the Table of Contents links work properly, it's best to avoid using LaTeX or special characters in headers. This helps GitHub generate correct anchor links.

3. **Ensuring LaTeX expressions are properly formatted**: GitHub supports LaTeX expressions enclosed in `$...$` for inline math and `$$...$$` for display math.

Below is the adjusted document:

---

# Protoplanetary Disk Simulation Toolkit Documentation

## Table of Contents

- [Introduction](#introduction)
- [Physical Setup](#physical-setup)
  - [Stellar Mass Distribution](#stellar-mass-distribution)
  - [Disk Mass and Structure](#disk-mass-and-structure)
  - [Temperature Profile](#temperature-profile)
  - [Aspect Ratio (H/R)](#aspect-ratio-hr)
  - [Dust Properties](#dust-properties)
  - [Cooling Parameter (beta_cool)](#cooling-parameter-betacool)
  - [Planetary System Generation](#planetary-system-generation)
- [Parameter Sampling Methods](#parameter-sampling-methods)
  - [Core and Tail Sampling](#core-and-tail-sampling)
  - [Statistical Distributions](#statistical-distributions)
- [References](#references)
- [Detailed Explanations and Additional Notes](#detailed-explanations-and-additional-notes)

---

## Introduction

PhantomBulk is designed to generate, run, and post-process simulations of protoplanetary disks (PPDs) using the **PHANTOM** and **MCFOST** software packages on high-performance computing (HPC) systems managed by SLURM. The simulations aim to produce physically realistic models of PPDs by sampling parameters based on empirical distributions from astronomical observations and theoretical considerations. This documentation provides an in-depth explanation of the physics involved, the methods used for parameter selection, and the rationale behind the choices made in the scripts.

---

## Physical Setup

The physical setup of the simulations is grounded in astrophysical observations and theoretical models of protoplanetary disks and star formation. Below is a detailed explanation of each aspect of the physical setup, including the limits and statistical methods used for choosing parameters.

### Stellar Mass Distribution

**Purpose**: To generate stellar masses for the simulations that reflect the observed distribution of stellar masses in the galaxy.

**Method**:

- **Initial Mass Function (IMF)**: The stellar masses are sampled based on the **Kroupa IMF** [1], which is a broken power-law distribution commonly used to represent the mass distribution of stars formed in a cluster.

  The IMF is defined as:

  $$
  \xi(M) \propto
  \begin{cases}
    M^{-1.3}, & \text{if } M < 0.5\,M_\odot \\
    M^{-2.3}, & \text{if } M \geq 0.5\,M_\odot
  \end{cases}
  $$

**Implementation**:

- The script samples stellar masses between **0.1 and 5 solar masses ($M_\odot$)**, reflecting the typical range of stellar masses for pre-main-sequence stars hosting protoplanetary disks.
- The sampling distinguishes between low-mass and high-mass segments based on the mass break at **0.5 $M_\odot$**.

**References**:

- [1] Kroupa, P. (2001). "On the variation of the initial mass function." *Monthly Notices of the Royal Astronomical Society*, 322(2), 231-246.

### Disk Mass and Structure

**Purpose**: To assign disk masses and structural parameters that are physically consistent with observations of protoplanetary disks.

**Method**:

- **Disk-to-Star Mass Ratio**: The disk mass is sampled as a fraction of the stellar mass. Observations suggest a typical disk-to-star mass ratio of around **1%** [2][3].
- **Surface Density Profile**: The surface density $\Sigma(R)$ follows a power-law profile:

  $$
  \Sigma(R) = \Sigma_0 \left( \frac{R}{R_{\text{ref}}} \right)^{-p}
  $$

  where $p$ is the surface density power-law index, and $R_{\text{ref}}$ is a reference radius (usually 1 AU).

**Parameter Ranges**:

- **Disk Mass Fraction**:
  - **Core Range**: 0.5% to 5% of stellar mass.
  - **Tail Range**: 0.1% to 10% of stellar mass.
- **Surface Density Power-law Index ($p$)**:
  - Typical values between **0.5 and 1.5** [2].
- **Inner Radius ($R_{\text{in}}$)**:
  - **Core Range**: 0.1 to 1 AU.
  - **Tail Range**: 0.05 to 5 AU.
- **Outer Radius ($R_{\text{out}}$)**:
  - **Core Range**: 30 to 100 AU.
  - **Tail Range**: 20 to 300 AU.

**Implementation**:

- The disk mass is calculated as:

  $$
  M_{\text{disk}} = f_{\text{disk}} \times M_\ast
  $$

  where $f_{\text{disk}}$ is the disk-to-star mass ratio.

- $\Sigma_0$ is determined by integrating the surface density profile over the disk area to match the total disk mass.

**References**:

- [2] Andrews, S. M., et al. (2010). "Protoplanetary Disk Structures in Ophiuchus." *The Astrophysical Journal*, 723(2), 1241.
- [3] Ansdell, M., et al. (2016). "ALMA Survey of Lupus Protoplanetary Disks I: Dust and Gas Masses." *The Astrophysical Journal*, 828(1), 46.

### Temperature Profile

**Purpose**: To define the thermal structure of the disk, which affects its scale height and stability.

**Method**:

- The temperature profile follows a power-law:

  $$
  T(R) = T_0 \left( \frac{R}{R_{\text{ref}}} \right)^{q}
  $$

  where $T_0$ is the temperature at the reference radius $R_{\text{ref}}$, and $q$ is the temperature power-law index.

**Parameter Ranges**:

- **Temperature at 1 AU ($T_0$)**:
  - Mean value adjusted based on stellar mass.
  - **Mean (log)**: $\ln(300\,\text{K})$.
  - **Standard Deviation (log)**: 0.2.
  - **Minimum**: 150 K.

- **Temperature Power-law Index ($q$)**:
  - **Mean**: $-0.5$.
  - **Standard Deviation**: 0.1.

**Implementation**:

- The temperature at 1 AU increases with stellar mass, reflecting that more massive stars tend to have hotter disks.
- Sampling ensures that temperatures remain within physically plausible ranges.

### Aspect Ratio (H/R)

**Purpose**: To determine the disk's vertical structure, which is critical for understanding its stability and evolution.

**Method**:

- The aspect ratio \(H/R\) is related to the sound speed \(c_s\) and the Keplerian orbital velocity \(v_{\text{orb}}\):

  $$
  \frac{H}{R} = \frac{c_s}{v_{\text{orb}}}
  $$

- The sound speed \(c_s\) depends on the temperature:

  $$
  c_s = \sqrt{\frac{k_B T}{\mu m_H}}
  $$

  where \(k_B\) is the Boltzmann constant, \(\mu\) is the mean molecular weight, and \(m_H\) is the mass of a hydrogen atom.

**Parameter Ranges**:

- **Aspect Ratio ($H/R$)**:
  - **Core Range**: 0.08 to 0.15.
  - **Tail Range**: 0.05 to 0.25.
- **Mean Molecular Weight ($\mu$)**: 2.34 (typical for molecular hydrogen).

**Implementation**:

- The aspect ratio is calculated based on the computed temperature profile and stellar mass.
- Limits are set to ensure that the disk is neither too thin (which could lead to numerical issues) nor too thick (which may not represent typical disks).

### Dust Properties

**Purpose**: To include the effects of dust in the simulations, which are crucial for radiative transfer and observational predictions.

**Parameters**:

- **Dust-to-Gas Mass Ratio**:
  - **Core Range**: 0.01 to 0.02.
  - **Tail Range**: 0.005 to 0.03.
- **Grain Size**:
  - **Core Range**: 0.001 to 0.1 mm.
  - **Tail Range**: 0.001 to 1.0 mm.
- **Grain Density**:
  - **Core Range**: 2.0 to 3.5 g/cm³.
  - **Tail Range**: 1.5 to 3.5 g/cm³.

**Implementation**:

- The dust-to-gas ratio is set close to the interstellar medium (ISM) value of 1%.
- Grain sizes and densities are sampled to represent typical dust grains found in protoplanetary disks.

### Cooling Parameter (beta_cool)

**Purpose**: To model the disk's thermal evolution by setting the cooling timescale.

**Method**:

- The **$\beta$-cooling** prescription from **Gammie (2001)** [4] is used:

  $$
  t_{\text{cool}} = \beta_{\text{cool}}\, \Omega^{-1}
  $$

  where \(t_{\text{cool}}\) is the cooling timescale, \(\beta_{\text{cool}}\) is a dimensionless cooling parameter, and \(\Omega\) is the orbital angular frequency.

**Parameter Ranges**:

- **Cooling Parameter ($\beta_{\text{cool}}$)**:
  - **Core Range**: 30 to 50.
  - **Tail Range**: 20 to 100.
  - **Minimum Value**: 30 (to prevent excessive cooling that could lead to artificial fragmentation).

**Implementation**:

- Sampling ensures that the cooling timescale is long enough to avoid artificial gravitational instabilities due to rapid cooling.

**References**:

- [4] Gammie, C. F. (2001). "Nonlinear Outcome of Gravitational Instability in Cooling, Gaseous Disks." *The Astrophysical Journal*, 553(1), 174-183.

### Planetary System Generation

**Purpose**: To include planets in the simulations and study their interactions with the disk.

**Method**:

- The number of planets and their properties are determined based on disk mass and other physical considerations.
- **Maximum Number of Planets**: Limited to 4 to reflect observed exoplanetary systems and to manage computational resources.
- **Planet Masses**: Sampled based on disk mass and distance from the star, considering isolation mass and Hill sphere constraints.

**Parameter Ranges**:

- **Planet Mass**:
  - Minimum: 0.1 Jupiter masses ($M_{\text{Jup}}$).
  - Maximum: 10 $M_{\text{Jup}}$.
- **Orbital Radius**:
  - Within the disk boundaries, avoiding the immediate vicinity of $R_{\text{in}}$ and $R_{\text{out}}$.
- **Inclination**:
  - Sampled from a Rayleigh distribution with a maximum of 15 degrees.

**Implementation**:

- Planets are placed at randomly selected radii within the disk, ensuring they are well within the disk boundaries.
- Accretion radii and J2 moments are assigned to simulate planetary accretion and oblateness.

---

## Parameter Sampling Methods

### Core and Tail Sampling

**Purpose**: To capture both the typical (core) values of parameters and less common (tail) values to explore a wider parameter space.

**Method**:

- **Core Range**: Represents the most probable values based on observations.
- **Tail Range**: Extends into less probable values, allowing for the exploration of extreme cases.
- **Tail Probability**: Set to 5%, meaning there is a 5% chance to sample from the tail range.

**Implementation**:

- A uniform random number determines whether to sample from the core or tail range.
- This approach ensures that most simulations reflect typical conditions, while a minority explore the extremes.

### Statistical Distributions

**Stellar Mass**:

- **Broken Power-law**: Based on the Kroupa IMF, with different slopes for low-mass and high-mass segments.

**Disk-to-Star Mass Ratio**:

- **Log-normal Distribution**: The logarithm of the disk-to-star mass ratio is normally distributed.

**Temperature and Aspect Ratio**:

- **Normal Distribution**: For the logarithm of $T_0$ and $q$, reflecting the scatter observed in disk temperatures.

**Planetary Parameters**:

- **Uniform Distribution**: For planetary masses and orbital radii within specified limits.
- **Rayleigh Distribution**: For orbital inclinations, representing a common distribution for small, positive values.

---

## References

1. **Kroupa Initial Mass Function**:
   - Kroupa, P. (2001). "On the variation of the initial mass function." *Monthly Notices of the Royal Astronomical Society*, 322(2), 231-246.

2. **Disk Properties and Mass Distributions**:
   - Andrews, S. M., et al. (2010). "Protoplanetary Disk Structures in Ophiuchus." *The Astrophysical Journal*, 723(2), 1241.
   - Ansdell, M., et al. (2016). "ALMA Survey of Lupus Protoplanetary Disks I: Dust and Gas Masses." *The Astrophysical Journal*, 828(1), 46.

3. **Cooling Parameter and Disk Stability**:
   - Gammie, C. F. (2001). "Nonlinear Outcome of Gravitational Instability in Cooling, Gaseous Disks." *The Astrophysical Journal*, 553(1), 174-183.

4. **Temperature Profiles in Disks**:
   - Dullemond, C. P., & Monnier, J. D. (2010). "The Inner Regions of Protoplanetary Disks." *Annual Review of Astronomy and Astrophysics*, 48, 205-239.

5. **Planet Formation Theories**:
   - Mordasini, C., et al. (2012). "Extrasolar Planet Population Synthesis I: Method, Formation Tracks, and Mass-Distance Distribution." *Astronomy & Astrophysics*, 541, A97.

---

**Note**: Replace placeholder paths, emails, and usernames in the scripts and documentation with your actual details. Always ensure compliance with your HPC system's policies and guidelines when running simulations.

---

## Detailed Explanations and Additional Notes

### Physical Constraints and Validation

- **Disk Stability**: The simulations ensure that the Toomre $Q$ parameter remains above critical values to avoid unphysical fragmentation. The aspect ratio and cooling parameters are adjusted accordingly.

- **Parameter Correlations**: Certain parameters are not independently sampled but are correlated based on physical laws. For example, the aspect ratio $H/R$ is computed from the temperature profile and stellar mass.

### Computational Considerations

- **Limitations on Planet Numbers**: The maximum number of planets is limited to manage computational resources and reflect observed planetary systems.

- **Execution of External Software**: The scripts assume that PHANTOM and MCFOST are properly installed and accessible. Users may need to adjust paths and environment variables.

### Statistical Sampling

- **Random Seed**: A seed is set (`seed=42`) for reproducibility. Users can change or remove the seed for different random samples.

- **Rejection Sampling**: If generated parameters fail validation, the script retries up to a maximum number of attempts to prevent infinite loops.

### Extending the Toolkit

- Users can extend the parameter ranges or modify the sampling methods to explore different regions of the parameter space.

- Additional physics, such as magnetic fields or radiation pressure, can be incorporated by modifying the setup templates and parameter generation methods.