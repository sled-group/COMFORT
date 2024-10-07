
import os
import sys
sys.path.append(os.getcwd())

import copy
import json
import argparse

import cv2
import numpy as np

from data_generation.utils import render_scene_config

from data_generation.constants import *
from data_generation.custom_variations import custom_variations
from data_generation.comfort_ball_config import *
from data_generation.comfort_car_left_config import *
from data_generation.comfort_car_right_config import *

def parse_args():
    parser = argparse.ArgumentParser(description="Render scene configuration script")
    parser.add_argument(
        '--dataset_name', type=str, required=True, 
        choices=['comfort_ball', 'comfort_car_left', 'comfort_car_right'],
        help="Dataset name to specify the configuration (comfort_ball, comfort_car_left, comfort_car_right)"
    )
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the rendered images")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "comfort_ball":
        default_config = COMFORT_BALL_DEFAULT_CONFIG
        variations = COMFORT_BALL_VARIATIONS
        relations = COMFORT_BALL_RELATIONS
        dataset_name = "comfort_ball"
    elif args.dataset_name == "comfort_car_left":
        default_config = COMFORT_LEFT_DEFAULT_CONFIG
        variations = COMFORT_LEFT_VARIATIONS
        relations = COMFORT_LEFT_RELATIONS
        dataset_name = "comfort_car_ref_facing_left"
    elif args.dataset_name == "comfort_car_right":
        default_config = COMFORT_RIGHT_DEFAULT_CONFIG
        variations = COMFORT_RIGHT_VARIATIONS
        relations = COMFORT_RIGHT_RELATIONS
        dataset_name = "comfort_car_ref_facing_right"
    else:
        raise ValueError("Invalid dataset name")
    
    default_config["save_path"] = os.path.join(args.save_path, dataset_name)

    for relation, relation_config in relations.items():
        distractors = []
        for variation in variations:
            if args.debug:
                variation["num_steps"] = 3

            relation_config_copy = copy.deepcopy(relation_config)
            relation_config_copy = custom_variations(relation, variation, default_config, relation_config_copy, dataset_name=args.dataset_name)
            config = {**default_config, **variation, **relation_config_copy}
            print(f"Variation: {variation['variation']} | Relation: {relation}", file=sys.stderr)

            mapping, distractors = render_scene_config(
                **config,
                distractors=distractors,
                dataset_name=args.dataset_name,
                render_shadow=True,
                cuda=False
            )

            output_path = os.path.join(args.save_path, dataset_name, relation, variation['variation'])

            if not args.debug:
                images = [cv2.imread(os.path.join(output_path, f'{i}.png')) for i in range(4)]
                indices = [0, 9, 18, 27]
                images = [cv2.imread(os.path.join(output_path, f'{I}.png')) for I in indices]
                height, width, channels = images[0].shape
                composite_image = np.zeros((height * 2, width * 2, channels), dtype=np.uint8)
                composite_image[0:height, 0:width] = images[0]       # Top-left
                composite_image[0:height, width:width*2] = images[1] # Top-right
                composite_image[height:height*2, 0:width] = images[2] # Bottom-left
                composite_image[height:height*2, width:width*2] = images[3] # Bottom-right
                cv2.imwrite(os.path.join(output_path, 'composite_image.png'), composite_image)
                print(f"Saved composite image to {os.path.join(output_path, 'composite_image.png')}", file=sys.stderr)


            os.makedirs(output_path, exist_ok=True)
            config_path = os.path.join(output_path, f"config.json")
            with open(config_path, 'w') as f:
                config_copy = copy.deepcopy(config)
                config_copy['var_color'] = color_to_name(config['var_color'])
                config_copy['ref_color'] = color_to_name(config['ref_color'])
                config_copy['mapping'] = mapping
                if distractors is not None and not isinstance(distractors, str):
                    for distractor in distractors:
                        distractor['location'] = np.array(distractor['location']).tolist()
                        distractor['dimensions'] = np.array(distractor['dimensions']).tolist()
                        distractor['color'] = color_to_name(distractor['color'])
                        distractor['position'] = np.array(distractor['position']).tolist()
                # print(distractors, file=sys.stderr)
                config_copy['distractor'] = distractors
                # assert(len(mapping) == DEFAULT_CONFIG["num_steps"])

                json.dump(config_copy, f, indent=4)                