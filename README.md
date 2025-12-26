# Sustainable AI Agents: Deep Layer-wise Structural Optimization for Energy-Efficient Recurrent Neural Networks

> **Abstract:** This research introduces a novel **Deep Layer-wise Reinforcement Learning (RL)** framework for automating the structural optimization of Gated Recurrent Unit (GRU) networks deployed on energy-constrained edge devices. Unlike traditional AutoML approaches (Bayesian Optimization, Evolutionary Strategies) that often converge to uniform pruning strategies, our proposed RL agent learns to exploit **layer-specific sensitivity**. By autonomously discovering a strategy that preserves feature extraction layers while aggressively compressing abstract representation layers, the proposed method achieves a **62.7% sparsity ratio** and a **63% reduction in FLOPs**, significantly outperforming state-of-the-art baselines while maintaining predictive accuracy within a 1% margin.

---

## 1. Introduction & Motivation

The deployment of Deep Learning models on Internet of Things (IoT) devices is hindered by severe constraints in energy, memory, and computational bandwidth. Traditional model compression techniques, such as **Uniform Pruning**, often fail to maximize efficiency because they treat all neural network layers as equally important.

Current **AutoML** solutions, such as **Tree-structured Parzen Estimator (TPE)** or **CMA-ES**, theoretically automate hyperparameter search. However, our empirical analysis reveals that these methods often stagnate at local optima (typically ~30% sparsity) because they lack the granular control required to differentiate between sensitive and redundant layers.

**This project proposes a "Deep Granular" approach:** treating the model compression problem as a multi-objective Markov Decision Process (MDP), where a PPO Agent learns to surgically prune specific layers to maximize energy efficiency without compromising model integrity.

---

## 2. Methodology

We established a rigorous comparative framework involving four distinct optimization strategies applied to a standard Weather Forecasting GRU architecture.

### 2.1. Baseline Architecture
* **Model:** 2-Layer Stacked GRU + Linear Readout.
* **Input:** Time-series weather data (Seattle Weather Dataset).
* **Precision:** FP32 (Full Precision).

### 2.2. Comparative Strategies
1.  **Manual Heuristic (Uniform):** Naive pruning applied uniformly across all layers.
2.  **Bayesian Optimization (TPE - AutoML):** A probabilistic model (Optuna) searching for the optimal global pruning rate and quantization flags.
3.  **Evolutionary Strategy (CMA-ES - AutoML):** A genetic algorithm approach to evolving model structure.
4.  **Proposed Method (Deep Layer-wise RL Agent):**
    * **Algorithm:** Proximal Policy Optimization (PPO) with Continuous Action Space.
    * **State Space ($S$):** Current Accuracy, Accuracy Drop ($\Delta Acc$), Parameter Reduction Ratio, FLOPs Reduction Ratio.
    * **Action Space ($A$):** A continuous vector $[a_0, a_1, a_2, a_3] \in [0, 1]^4$ controlling:
        * $a_0$: Pruning Rate for GRU Layer 0 (Feature Extraction).
        * $a_1$: Pruning Rate for GRU Layer 1 (Abstract Representation).
        * $a_2$: Pruning Rate for Linear Readout.
        * $a_3$: Probability of enabling Dynamic Quantization.
    * **Reward Function ($R$):** A composite function penalizing accuracy loss while rewarding FLOPs reduction.

---

## 3. Experimental Results

The following results were obtained from the final evaluation pipeline (`evaluate-benchmark.ipynb`), averaging metrics over $N=30$ independent inference runs to ensure statistical robustness.

### 3.1. Quantitative Comparison

| Optimization Method | Accuracy (%) | Sparsity (Compression) | Computational Cost (FLOPs) | Energy Efficiency (mWh) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP32)** | 65.85% | 0.0% | 35.97 M | 1.578 |
| **AutoML (TPE)** | 65.16% | 29.8% | 25.24 M | 1.459 |
| **AutoML (CMA-ES)** | 65.16% | 29.8% | 25.24 M | 1.478 |
| **RL Agent (Ours)** | **64.81%** | **62.7%** 🚀 | **13.42 M** 📉 | **1.489** |

### 3.2. Critical Analysis
* **The "Uniform Barrier":** Both AutoML methods (TPE and CMA-ES) converged to a sparsity of **~30%**. Analysis confirms they applied a uniform pruning rate. They could not prune deeper because applying high sparsity to the sensitive **Input-to-Hidden (Layer 0)** caused immediate accuracy collapse.
* **The RL Breakthrough:** The RL Agent discovered a non-intuitive **Global Optimum**:
    * **Layer 0 (Input):** Kept at **0% Pruning** (Preserving raw feature extraction).
    * **Layer 1 (Hidden):** Pruned at **95%** (Removing redundant abstract features).
    * **Result:** This allowed the model to achieve **2.1x higher compression** than AutoML while maintaining comparable accuracy (~1% tradeoff).

---

## 4. Repository Structure & Reproduction

This repository is organized into a scientific pipeline. To reproduce our findings, execute the notebooks in the following order:

| Step | Notebook File | Description | Output Artifact |
| :--- | :--- | :--- | :--- |
| **1** | `train-baseline.ipynb` | Preprocesses data and trains the unoptimized GRU model. | `baseline_model.pth`, `processed_data.pt` |
| **2** | `train-automl-baselines.ipynb` | Runs Optuna (TPE & CMA-ES) to establish SOTA benchmarks. | `automl_results.json` |
| **3** | `train-rl-agent.ipynb` | (Optional) Experimentation with discrete RL action spaces. | `best_action.json` |
| **4** | `train-rl-agent-expanded.ipynb` | **[CORE]** Trains the Deep Layer-wise PPO Agent using continuous control. | `best_action_expanded.json` |
| **5** | `evaluate-benchmark.ipynb` | Aggregates all models, calculates FLOPs/Energy, and generates Pareto plots. | `final_report_charts.png` |

---

## 5. Usage Guide

### Prerequisites
* Python 3.10+
* PyTorch (CUDA supported)
* Stable-Baselines3
* CodeCarbon

### Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/Sustainable_AI_Agents.git](https://github.com/your-username/Sustainable_AI_Agents.git)
cd Sustainable_AI_Agents

# Install dependencies
pip install -r requirements.txt
```

Running the Benchmark
Ensure all artifacts from Steps 1-4 are present in the working directory. Then, execute the evaluation script:

Bash

### In Jupyter/Kaggle environment
```bash
Run evaluate-benchmark.ipynb
```

## 6. Conclusion
This project demonstrates that structural granularity is the key to unlocking extreme model compression. By utilizing Deep Reinforcement Learning to navigate the complex trade-off between layer sensitivity and redundancy, we achieved a 63% reduction in computational cost. This methodology offers a viable path for deploying sophisticated recurrent neural networks on extreme-edge devices, contributing to the broader goal of Sustainable Green AI.