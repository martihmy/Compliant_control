# The Controllers

The folders "AC", "HFMC" and "VIC" contain the executable scripts, responsible for each compliant force controller.

- Admittance Control (AC)
- Hybrid Force/Motion Control (HFMC)
- Force-based Variable Impedance Control (VIC)

The controllers are implemented based on control laws from literary sources. The sources are documented in Compliant_control/LCRM_Figures/Master_Thesis_LCRM.pdf


# PILCO

## Recomandation

To best understand the PILCO-framework, I recommend to start by inspecting Compliant_control/gym-panda/HFMC/PILCO_HFMC.py and Compliant_control/gym-panda/HFMC/PILCO_HFMC_utils.py as these scripts contain more elaborate comments than those of AC and VIC.
