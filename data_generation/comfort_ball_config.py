import math
from data_generation.constants import *

COMFORT_BALL_DEFAULT_CONFIG = {
    "variation": "default",
    "name": None,
    "num_steps": 37,
    "save_path": None,

    "ref_shape": SPHERE,
    "ref_color": BLUE,
    "ref_size": 0.6,
    "ref_position": (0, 0, 0.6),
    "ref_rotation": (0, 0, 0),

    "var_shape": SPHERE,
    "var_color": RED,
    "var_size": 0.6,
    "var_position": (0, 0, 0.6),
    "var_rotation": (0, 0, 0),

    "num_distractors": 0,
    "cam_position": None,
}

COMFORT_BALL_RELATIONS = {
    BEHIND: {
        "relation": BEHIND,
        "path_type": "rotate",
        "radius": 2.9,
        "angle_range": (180, 180+360),
    },
    FRONT: {
        "relation": FRONT,
        "path_type": "rotate",
        "radius": 2.9,
        "angle_range": (0, 360),
    },
    LEFT: {
        "relation": LEFT,
        "path_type": "rotate",
        "radius": 2.9,
        "angle_range": (90, 90+360),
    },
    RIGHT: {
        "relation": RIGHT,
        "path_type": "rotate",
        "radius": 2.9,
        "angle_range": (270, 270+360),
    },
}

COMFORT_BALL_VARIATIONS = [
    {"variation": "default"},
    {"variation": "color", "var_color": YELLOW, "ref_color": GREEN},
    {"variation": "size", "var_size": 0.7, "ref_size": 0.45},
    {"variation": "cam_position", "cam_position": (9.0, 0.0, 7.0)},
    {"variation": "distractor", "num_distractors": 1},

]
