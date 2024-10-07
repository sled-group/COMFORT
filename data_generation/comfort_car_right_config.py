import math
from data_generation.constants import *

COMFORT_RIGHT_DEFAULT_CONFIG = {
    "num_distractors": 0,
    "name": None,
    "num_steps": 37
}

COMFORT_RIGHT_BASKETBALL_AND_CHAIR = {
    # render config
    "variation": "object_basketball_and_chair",
    "num_steps": 37,
    "radius": 3.3,

    # ref object config
    "ref_shape": CHAIR,
    "ref_color": "",
    "ref_size": 1.7,
    "ref_position": (0, 0, 0.9),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

COMFORT_RIGHT_BASKETBALL_AND_DOG = {
    # render config
    "variation": "object_basketball_and_dog",
    "num_steps": 37,
    "radius": 3,

    # ref object config
    "ref_shape": DOG,
    "ref_color": "",
    "ref_size": 1.0,
    "ref_position": (0, -0.6, 0.6),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

COMFORT_RIGHT_BASKETBALL_AND_SOFA = {
    # render config
    "variation": "object_basketball_and_sofa",
    "num_steps": 37,
    "radius": 3,

    # ref object config
    "ref_shape": COUCH,
    "ref_color": "",
    "ref_size": 1.3,
    "ref_position": (0, 0, 0.0),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

COMFORT_RIGHT_BASKETBALL_AND_BED = {
    # render config
    "variation": "object_basketball_and_bed",
    "num_steps": 37,
    "radius": 3.3,

    # ref object config
    "ref_shape": BED,
    "ref_color": "",
    "ref_size": 1.3,
    "ref_position": (0, 0, 0.5),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}


COMFORT_RIGHT_BASKETBALL_AND_LAPTOP = {
    # render config
    "variation": "object_basketball_and_laptop",
    "num_steps": 37,
    "radius": 3,

    # ref object config
    "ref_shape": LAPTOP,
    "ref_color": "",
    "ref_size": 4,
    "ref_position": (0, 0, 0.5),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}


COMFORT_RIGHT_BASKETBALL_AND_DUCK = {
    # render config
    "variation": "object_basketball_and_duck",
    "num_steps": 37,
    "radius": 3,

    # ref object config
    "ref_shape": DUCK,
    "ref_color": "",
    "ref_size": 5,
    "ref_position": (0, 0, 0.8),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}


COMFORT_RIGHT_BASKETBALL_AND_HORSE = {
    # render config
    "variation": "object_basketball_and_horse",
    "num_steps": 37,
    "radius": 3.5,

    # ref object config
    "ref_shape": HORSE_R,
    "ref_color": "",
    "ref_size": 1.1,
    "ref_position": (0, 0, 0),
    "ref_rotation": (0, 0, 0),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

COMFORT_RIGHT_BASKETBALL_AND_CAR = {
    # render config
    "variation": "default",
    "num_steps": 37,
    "radius": 3.2,

    # ref object config
    "ref_shape": CAR_SEDAN,
    "ref_color": AIRPLANE_WHITE,
    "ref_size": 2.2,
    "ref_position": (0, -0.1, 0.95),
    "ref_rotation": (90, 0, 360),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

COMFORT_RIGHT_BASKETBALL_AND_BENCH = {
    # render config
    "variation": "object_basketball_and_bench",
    "num_steps": 37,
    "radius": 3,

    # ref object config
    "ref_shape": BENCH,
    "ref_color": "",
    "ref_size": 1.25,
    "ref_position": (0, 0, 0.6),
    "ref_rotation": (0, 0, 180),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

COMFORT_RIGHT_BASKETBALL_AND_BICYCLE = {
    # render config
    "variation": "object_basketball_and_bicycle",
    "num_steps": 37,
    "radius": 3.1,

    # ref object config
    "ref_shape": BICYCLE_MOUNTAIN,
    "ref_color": DARK_GRAY,
    "ref_size": 1.8,
    "ref_position": (0.05, 0, 0.45),
    "ref_rotation": (90, 0, 90+360),

    # var object config
    "var_shape": BASKETBALL,
    "var_color": "",
    "var_size": 2,
    "var_position": (0, 0, 0.3),
    "var_rotation": (0, 0, 90),

    # addressee object config
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0, -5.0, 0.1),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 90),

    # variation config
    "num_distractors": 0,
    "cam_position": (10.0, 0, 9.0),    
}

# Positional attributes
COMFORT_RIGHT_RELATIONS = {
    # Rotation relations
    BEHIND: {
        "relation": BEHIND,
        "path_type": "rotate",
        "angle_range": (180, 180+360),
    },
    FRONT: {
        "relation": FRONT,
        "path_type": "rotate",
        "angle_range": (0, 360),
    },
    LEFT: {
        "relation": LEFT,
        "path_type": "rotate",
        "angle_range": (90, 90+360),
    },
    RIGHT: {
        "relation": RIGHT,
        "path_type": "rotate",
        "angle_range": (270, 270+360),
    },
}

COMFORT_RIGHT_VARIATIONS = [
    COMFORT_RIGHT_BASKETBALL_AND_HORSE,
    COMFORT_RIGHT_BASKETBALL_AND_BENCH,   
    COMFORT_RIGHT_BASKETBALL_AND_LAPTOP,  
    COMFORT_RIGHT_BASKETBALL_AND_DUCK,     
    COMFORT_RIGHT_BASKETBALL_AND_CHAIR,
    COMFORT_RIGHT_BASKETBALL_AND_DOG,     
    COMFORT_RIGHT_BASKETBALL_AND_SOFA,    
    COMFORT_RIGHT_BASKETBALL_AND_BED,
    COMFORT_RIGHT_BASKETBALL_AND_BICYCLE,   
    COMFORT_RIGHT_BASKETBALL_AND_CAR,
]