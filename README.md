# 🛋️ Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/)
[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)](https://spatial-comfort.github.io/)
[![Hugging Dataset](https://img.shields.io/badge/huggingface-dataset:COMFORT-green)](https://huggingface.co/datasets/sled-umich/COMFORT)

![COMFORT](comfort.jpg "COMFORT")

This repository provides the code and instructions for using the evaluation protocol to systematically assess the spatial reasoning capabilities of VLMs, <ins>CO</ins>nsistent <ins>M</ins>ultilingual <ins>F</ins>rame <ins>O</ins>f <ins>R</ins>eference <ins>T</ins>est (COMFORT). Follow the steps below to set up the environment, generate data (optional), and run experiments. Feel free to create an issue if you encounter any problems. We also welcome pull requests.

## Table of Contents

1. [Setup Environment](#setup-environment)
2. [Prepare Data](#prepare-data)
3. [Add API Credentials](#add-api-credentials)
4. [Run Experiments](#run-experiments)
5. [Run Evaluations](#run-evaluations)
6. [Evaluate More Models](#evaluate-more-models)
7. [Common Problems and Solutions](#common-problems-and-solutions)

## Setup Environment

Clone the repository and create a conda environment using the provided `environment.yml` file:


```bash
git clone https://github.com/sled-group/COMFORT.git
cd comfort_utils
conda env create -f environment.yml
```

After creating the environment:

```bash
conda activate comfort
```

Then, install editable packages:

```bash
cd models/GLAMM
pip install -e .
```

```bash
cd models/llava
pip install -e .
```

```bash
cd models/InternVL/internvl_chat
pip install -e .
```

You can also use [Poetry](https://python-poetry.org/docs/ "Poetry") to setup the environment.

## Prepare data

Firstly, make a data directory:
```bash
mkdir data
```

### (Option 1.) Download data from Huggingface
```bash
wget https://huggingface.co/datasets/sled-umich/COMFORT/resolve/main/comfort_ball.zip?download=true -O data/comfort_ball.zip
unzip data/comfort_ball.zip -d data/
wget https://huggingface.co/datasets/sled-umich/COMFORT/resolve/main/comfort_car_ref_facing_left.zip?download=true -O data/comfort_car_ref_facing_left.zip
unzip data/comfort_car_ref_facing_left.zip -d data/
wget https://huggingface.co/datasets/sled-umich/COMFORT/resolve/main/comfort_car_ref_facing_right.zip?download=true -O data/comfort_car_ref_facing_right.zip
unzip data/comfort_car_ref_facing_right.zip -d data/
```

### (Option 2.) Data generation
```bash
pip install gdown
python download_assets.py
chmod +x generate_dataset.sh
./generate_dataset.sh
```

## Add API Credentials
```bash
touch comfort_utils/model_utils/api_keys.py
```
1. Prepare OpenAI and DeepL API keys and add below to api_keys.py

    ```
    APIKEY_OPENAI = <YOUR_API_KEY>
    APIKEY_DEEPL = <YOUR_API_KEY>
    ```
2. Prepare Google Cloud Translate API credentials (.json)

## Run Experiments
```bash
./run_english_ball_experiments.sh
./run_english_car_left_experiments.sh
./run_english_car_right_experiments.sh

export GOOGLE_APPLICATION_CREDENTIALS="your_google_application_credentials_path.json"
./run_multilingual_ball_experiments.sh
./run_multilingual_car_left_experiments.sh
./run_multilingual_car_right_experiments.sh
```

## Run Evaluations
### English
1. Preferred Coordinate Transformation (Table 2 & Table 7):
    ```bash
    python gather_results.py --mode cpp --cpp convention
    ```
2. Preferred Frame of Reference (Table 3 & Table 8):
    ```bash
    python gather_results.py --mode cpp --cpp preferredfor
    ```
3. Perspective Taking (Table 4 & Table 9):
    ```bash
    python gather_results.py --mode cpp --cpp perspective
    ```
4. Comprehensive Evaluation (Table 5):
    ```bash
    python gather_results.py --mode comprehensive
    ```
### Multilingual (Figure 8 & Table 10)
```bash
python gather_results_multilingual.py
```

After evaluation completes:
```bash
cd results/eval
python eval_multilingual_preferredfor_raw.py
```


## Evaluate More Models
We refer to [Model Wrapper](comfort_utils/model_utils/wrapper.py "wrapper").

## Common Problems and Solutions
1. ImportError: libcupti.so.11.7: cannot open shared object file: No such file or directory
    ```bash
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    ```