# Sustainable AI Agents: Optimizing GRU Models using Reinforcement Learning

This repository contains the code for the research project investigating the use of Reinforcement Learning (RL), specifically Proximal Policy Optimization (PPO), to automate the optimization (pruning and quantization) of Gated Recurrent Unit (GRU) models for weather forecasting, with a focus on improving sustainability metrics like energy consumption.

## Overview

Artificial Intelligence models, while powerful, often demand significant computational resources, leading to high energy consumption and environmental impact. This project explores applying RL to find optimal configurations for GRU models that balance predictive accuracy with resource efficiency (FLOPs, energy). The PPO agent learns to select combinations of L1 magnitude pruning and dynamic quantization to achieve these Green AI goals. The code demonstrates significant theoretical reductions in FLOPs/parameters and measurable inference energy savings while maintaining baseline accuracy on the Seattle weather dataset.

## Dataset

* **Source:** Seattle Weather dataset from Kaggle.
* **Link:** [https://www.kaggle.com/datasets/ananthrtr/seattle-weather](https://www.kaggle.com/datasets/ananthrtr/seattle-weather)
* **Usage:** Please download the `seattle-weather.csv` file from the link above. When running the notebooks on Kaggle, add this dataset via the "+ Add Data" button and ensure the path in `train_baseline.ipynb` (within the `Config` class) points to `/kaggle/input/seattle-weather/seattle-weather.csv` (or the appropriate path provided by Kaggle). **Do not commit the `.csv` file to this repository.**

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/trongjhuongwr/Sustainable_AI_Agent_Project.git](https://github.com/trongjhuongwr/Sustainable_AI_Agent_Project.git)
    cd Sustainable_AI_Agent_Project
    ```
2.  **(Optional) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    Ensure you have Python 3.11 or compatible installed. It is recommended to run this in an environment with CUDA support (like Kaggle GPU notebooks) for faster training. PyTorch version with CUDA 12.1 is specified.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If running locally without CUDA 12.1, you might need to adjust the torch installation command or install the CPU version).*

## Usage / How to Run

The project is divided into three main Jupyter notebooks, designed to be run sequentially, preferably on a platform like Kaggle with GPU support:

1.  **`train_baseline.ipynb`:**
    * Loads and preprocesses the Seattle weather data.
    * Defines the `WeatherGRU` model architecture.
    * Trains the baseline GRU model using AdamW and Cosine Annealing scheduler.
    * Saves the processed data tensors to `processed_data.pt`.
    * Saves the trained baseline model weights to `baseline_model.pth`.
2.  **`train_rl_agent.ipynb`:**
    * Loads the baseline model (`baseline_model.pth`) and processed test data (`processed_data.pt`).
    * Defines helper functions and the custom RL environment (`SustainableAIAgentEnv`).
    * Initializes or loads a PPO agent.
    * Trains the PPO agent to learn the optimal pruning/quantization strategy.
    * Saves the trained agent to `sustainable_ai_agent_ppo.zip`.
    * (Optional output) Predicts and saves the best action found by the agent to `best_action.json`.
3.  **`evaluate_benchmark.ipynb`:**
    * Loads the baseline model (`baseline_model.pth`) and processed test data (`processed_data.pt`).
    * Loads the trained PPO agent (`sustainable_ai_agent_ppo.zip`) OR the best action (`best_action.json`).
    * Applies different optimization strategies (Baseline, Manual Pruning, Manual Quantization, Agent Optimized).
    * Runs a comprehensive benchmark using `evaluate_model` (including CodeCarbon for energy).
    * Displays the benchmark results table and generates comparison plots.

**Running on Kaggle:**

* Upload the three `.ipynb` files to a new Kaggle notebook.
* Add the "Seattle Weather" dataset via "+ Add Data".
* Ensure the file paths in the notebooks (especially for data loading/saving and model/agent loading/saving) are correct for the Kaggle environment (usually `/kaggle/working/` for outputs).
* Run the cells sequentially in each notebook, starting with `train_baseline.ipynb`.

## Results Summary

The RL agent identified **20% L1 Pruning + Dynamic Quantization** as the optimal strategy. The benchmark results showed:

* **Accuracy:** Maintained identical accuracy to the baseline (0.679443).
* **Parameters & FLOPs:** Significant theoretical reduction (reported near zero due to measurement tool limitations with quantization).
* **Energy Consumption:** Measurable reduction of ~3.75% during inference compared to the baseline.

| Model Version            | Accuracy | Params (M) | FLOPs (G) | Energy (mWh) | CO2 eq (kg) |
| :----------------------- | :------- | :--------- | :-------- | :----------- | :---------- |
| Baseline                 | 0.679443 | 0.596225   | 0.035974  | 0.002690     | 0.000001    |
| Manual Pruning (50%)     | 0.679443 | 0.596225   | 0.035974  | 0.002696     | 0.000001    |
| Manual Quantization      | 0.679443 | 0.000000\* | 0.000000\*| 0.002549     | 0.000001    |
| Agent Optimized (P20+Q) | 0.679443 | 0.000000\* | 0.000000\*| 0.002589     | 0.000001    |

*\*Note: Params/FLOPs reported as 0.0 for quantized models due to tool limitations.*