#!/bin/bash

# COMFORT BALL
python data_generation/generate_dataset.py --dataset_name comfort_ball --save_path ./data 1> /dev/null

# COMFORT CAR LEFT
python data_generation/generate_dataset.py --dataset_name comfort_car_left --save_path ./data 1> /dev/null

# COMFORT CAR RIGHT
python data_generation/generate_dataset.py --dataset_name comfort_car_right --save_path ./data 1> /dev/null