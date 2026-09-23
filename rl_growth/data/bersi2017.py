"""Experimental data of Bersi et al. (2017), murine infrarenal abdominal aorta, 28-day AngII infusion.

Values are APPROXIMATE readings of Figures 6 and 8 of Liu et al. (2022) (the original tables are not
reproduced in the RL paper).  Day-0 geometry, day-28 homeostatic stresses and day-0 pressure are the
exact values quoted in the text / Table 2.
"""
import numpy as np

days = np.array([0, 4, 7, 14, 21, 28], float)
systolic_pressure_mmHg = np.array([111.6, 155.0, 166.0, 181.0, 184.0, 186.0])   # Fig. 6(a)
unloaded_thickness_mm = np.array([0.099, 0.114, 0.128, 0.148, 0.147, 0.152])    # Fig. 6(b)
systolic_inner_radius_mm = np.array([0.418, 0.450, 0.455, 0.448, 0.460, 0.464]) # Fig. 6(c)
in_vivo_axial_stretch = np.array([1.81, 1.79, 1.77, 1.72, 1.69, 1.65])          # Fig. 6(d)
sigma_tt_kPa = np.array([194.0, 250.0, 245.0, 205.0, 215.0, 221.0])             # Fig. 8(a)
sigma_zz_kPa = np.array([258.0, 280.0, 275.0, 233.0, 225.0, 226.2])             # Fig. 8(b)
