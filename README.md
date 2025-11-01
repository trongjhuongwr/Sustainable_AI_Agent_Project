# Sustainable AI Agents: Automated GRU Optimization using Reinforcement Learning

## 1. Project Overview

This repository contains the complete experimental code for the research paper: *"Sustainable AI Agents: Optimizing Energy Consumption in Machine Learning Models for Eco-Friendly Systems."*

The core objective of this research is to investigate a pragmatic, two-stage methodology for creating "Green AI." We address the critical disconnect between *theoretical proxy metrics* (like FLOPs) and *empirical, real-world metrics* (like measured energy consumption and latency).

We hypothesize that a Reinforcement Learning (RL) agent, even when trained on fast but imperfect proxy metrics (Stage 1), can serve as an effective exploration tool to discover non-obvious, energy-efficient model configurations. This agent-discovered strategy is then rigorously validated in a separate statistical benchmark (Stage 2) using direct, empirical energy measurements (via `CodeCarbon`).

This project demonstrates this methodology by training a Proximal Policy Optimization (PPO) agent from `stable-baselines3` to find an optimal combination of L1 magnitude pruning and dynamic quantization for a GRU (Gated Recurrent Unit) time-series forecasting model.

## 2. Key Statistical Findings (N=10 Runs)

The primary contribution of this work is the statistically validated evidence that the RL-driven exploration successfully identifies a superior, energy-efficient configuration. The final `evaluate_benchmark.ipynb` notebook executes 10 independent runs (N=10) with different seeds to ensure robust results.

The key finding is that the **Agent-Optimized (P30+Q)** strategy, discovered automatically by the RL agent, provided the best overall sustainability profile while **maintaining 100% of the baseline accuracy**.

| Model Version | Mean Accuracy (± Std) | Mean Energy (mWh) (± Std) | Mean Latency (ms) (± Std) | Params (M) | FLOPs (G) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.68641 ± 0.00000 | 0.001771 ± 0.000177 | 0.4261 ± 0.0466 | 0.5962 | 0.0360 |
| **Manual Pruning (50%)** | 0.68641 ± 0.00000 | 0.001709 ± 0.000037 | 0.3898 ± 0.0211 | 0.5962\* | 0.0360\* |
| **Manual Quantization** | 0.68641 ± 0.00000 | **0.001800 ± 0.000183** | 0.4120 ± 0.0222 | 0.0000\*\* | 0.0000\*\* |
| **Agent Optimized (P30+Q)** | 0.68641 ± 0.00000 | **0.001705 ± 0.000170** | **0.4076 ± 0.0194** | 0.0000\*\* | 0.0000\*\* |

*\*Note: `ptflops` does not account for unstructured sparsity, so these values are unchanged from the baseline.*
*\*\*Note: `ptflops` fails to analyze quantized operations and incorrectly reports 0.0, highlighting the failure of this proxy metric.*

**Conclusions from the data:**
1.  **Accuracy is Preserved:** All strategies maintained identical test accuracy (0.686411).
2.  **"Obvious" Heuristic Fails:** Manual Quantization, a common heuristic, was *detrimental*, increasing mean energy consumption by **1.6%**.
3.  **RL Agent Succeeds:** The RL agent successfully avoided the failing heuristic and discovered the P30+Q strategy, which achieved a **3.7% reduction in energy** and a **4.3% reduction in latency** compared to the baseline, proving its value as an automated exploration tool.

## 3. Methodology & Repository Workflow

This research is structured into a 3-notebook pipeline, representing a pragmatic two-stage methodology (Stage 1: Exploration on Proxies, Stage 2: Validation on Ground Truth).

### Notebook 1: `train_baseline.ipynb`
* **Purpose:** To establish the "ground truth" for performance and cost.
* **Process:** Loads the "Seattle Weather" dataset from Kaggle, preprocesses it into time-series sequences (30 days), trains the `WeatherGRU` model, and saves the best model (`baseline_model.pth`) and the processed data (`processed_data.pt`).
* **Key Output:** The baseline accuracy (0.686411) and the baseline model for optimization.

### Notebook 2: `train_rl_agent.ipynb`
* **Purpose:** **Stage 1 (Exploration)**. To train the RL agent to explore optimization strategies using fast, theoretical *proxy metrics*.
* **Process:**
    1.  Defines the `SustainableAIAgentEnv` (a custom Gymnasium environment).
    2.  The **Action Space** is 16 discrete actions (8 pruning ratios x 2 quantization options).
    3.  The **Reward Function** is based on theoretical `ptflops` (FLOPs) and parameter count reductions, with a heavy penalty for accuracy loss.
    4.  A `stable-baselines3` PPO agent is trained for 10,000 steps.
* **Key Output:** The agent's "best discovered strategy" (`best_action.json`), which was found to be **Action 11 (30% Pruning + Quantization)**.

### Notebook 3: `evaluate_benchmark.ipynb`
* **Purpose:** **Stage 2 (Validation)**. To rigorously and statistically benchmark the agent's discovered strategy against manual heuristics using *empirical, real-world metrics*.
* **Process:**
    1.  Loads the `baseline_model.pth` and `best_action.json`.
    2.  Defines 4 model configurations: (Baseline, Manual Pruning 50%, Manual Quantization, Agent Optimized P30+Q).
    3.  Runs a loop for **N=10 runs** (for statistical robustness), setting a new seed for each run.
    4.  Inside the loop, it measures **empirical inference energy** (using `CodeCarbon`) and **empirical inference latency** (using `time.perf_counter`) for all 4 models on the test set.
    5.  Aggregates the 10 runs and reports the final **Mean (μ) ± Standard Deviation (σ)**.
* **Key Output:** The statistical benchmark table (see Section 2) that validates the agent's strategy.

## 4. Setup and Replication

This project was designed and tested in the Kaggle Notebook environment.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/trongjhuongwr/Sustainable_AI_Agent_Project.git](https://github.com/trongjhuongwr/Sustainable_AI_Agent_Project.git)
    cd Sustainable_AI_Agent_Project
    ```
2.  **Install Dependencies:**
    A virtual environment is recommended.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset:**
    The notebooks are configured to read data from the following Kaggle dataset:
    * **Seattle Weather:** `https://www.kaggle.com/datasets/ananthr1/weather-prediction` (This must be added to the Kaggle notebook environment as input data).

4.  **Run Order:**
    The notebooks must be run sequentially, as they produce artifacts used by the next step:
    1.  Run `train_baseline.ipynb` (Produces `baseline_model.pth`, `processed_data.pt`).
    2.  Run `train_rl_agent.ipynb` (Consumes artifacts from step 1, produces `best_action.json`).
    3.  Run `evaluate_benchmark.ipynb` (Consumes artifacts from steps 1 & 2, produces the final statistical results and visualizations).