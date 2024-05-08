import os
from pathlib import Path

# --- define paths applied in the repo
sb_excerpt_dir = os.path.dirname(os.path.realpath(__file__))
home = str(Path.home())
data_path = os.path.join(sb_excerpt_dir, "data")
profiles_file = os.path.join(os.path.dirname(data_path), "data", "profiles.h5")

# --- import functionality
import SimBench_EHV_HV_excerpt.toolbox

from .SimBench_for_phd import SimBench_for_phd

from .grid_parameters import SimBench_for_phd_obj_weights, grid_parameters

from .data_overview import electric_params, element_numbers
