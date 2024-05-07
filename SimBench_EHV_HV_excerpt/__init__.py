import os
sb_excerpt_dir = os.path.dirname(os.path.realpath(__file__))

import SimBench_EHV_HV_excerpt.toolbox

from .SimBench_for_phd import SimBench_for_phd

from .grid_parameters import SimBench_for_phd_obj_weights, grid_parameters

from .data_overview import electric_params, element_numbers
