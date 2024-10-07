import sys
from data_generation.constants import *

def custom_variations(relation, variation, default_config, relation_config_copy, dataset_name=None):
    if dataset_name == "comfort_ball":
        if relation in ROTATION_LIST:
            if variation['variation'] == "size":
                relation_config_copy['ref_position'] = (0, 0, variation['ref_size'])
                relation_config_copy['var_position'] = (0, 0, variation['var_size'])
    return relation_config_copy
