import torch
import pandas as pd
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

from src.config import Config
from src.model import initialize_model
from src.data_processing import load_data
from src.evaluation import evaluate_model
from src.optimization import apply_l1_pruning, apply_dynamic_quantization
from src.environment import SustainableAIAgentEnv

def run_full_benchmark(config: Config, baseline_model: nn.Module, test_loader: torch.utils.data.DataLoader):
    """
    Runs a comprehensive benchmark comparing different optimization methods.

    Args:
        config (Config): Configuration object.
        baseline_model (nn.Module): The trained baseline model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.

    Returns:
        pd.DataFrame: DataFrame containing the benchmark results. Returns empty DataFrame on critical error.
    """
    results = {}
    best_action_info = None

    # --- Get Best Action from Trained Agent ---
    try:
        if not os.path.exists(config.AGENT_SAVE_PATH):
            print(f"Warning: Agent file not found at {config.AGENT_SAVE_PATH}. Cannot evaluate Agent Optimized model.")
        else:
            print("\n--- Loading trained agent to determine best action ---")
            # Need a temporary env instance to load the agent
            temp_env = SustainableAIAgentEnv(baseline_model, test_loader, config)
            agent = PPO.load(config.AGENT_SAVE_PATH, env=temp_env)
            print("Agent loaded.")

            # Get initial observation from the env to predict action
            obs, _ = temp_env.reset()
            # Use deterministic=True for the best known action
            action, _ = agent.predict(obs, deterministic=True)
            action_item = action.item()
            print(f"Agent predicts best action: {action_item}")

            # Decode action
            num_pruning_levels = len(temp_env.pruning_levels)
            pruning_index = action_item % num_pruning_levels
            pruning_amount = temp_env.pruning_levels[pruning_index]
            apply_quant = (action_item >= num_pruning_levels)

            best_action_info = {
                'pruning_amount': pruning_amount,
                'quantized': apply_quant
            }
            del temp_env # Clean up temporary environment
            del agent # Clean up loaded agent

    except Exception as e:
        print(f"Error loading or predicting from agent: {e}. Cannot evaluate Agent Optimized model.")
        best_action_info = None # Ensure it's None

    # --- Run Evaluations ---
    # Ensure baseline is on CPU for consistent comparisons
    baseline_model_cpu = copy.deepcopy(baseline_model).cpu()

    print("\n--- 1. Evaluating Baseline Model ---")
    results['Baseline'] = evaluate_model(baseline_model_cpu, test_loader, torch.device('cpu'), config)

    print("\n--- 2. Evaluating Manual Pruning (50%) ---")
    try:
        # Start from a fresh CPU copy for each method
        pruned_model = apply_l1_pruning(copy.deepcopy(baseline_model_cpu), 0.5, config)
        results['Manual Pruning (50%)'] = evaluate_model(pruned_model, test_loader, torch.device('cpu'), config)
    except Exception as e:
        print(f"Error during Manual Pruning evaluation: {e}")
        results['Manual Pruning (50%)'] = {} # Store empty dict on error

    print("\n--- 3. Evaluating Manual Quantization ---")
    try:
        quantized_model = apply_dynamic_quantization(copy.deepcopy(baseline_model_cpu))
        results['Manual Quantization'] = evaluate_model(quantized_model, test_loader, torch.device('cpu'), config)
    except Exception as e:
        print(f"Error during Manual Quantization evaluation: {e}")
        results['Manual Quantization'] = {}

    print("\n--- 4. Evaluating Agent-Optimized Model ---")
    if best_action_info is not None:
        pruning_amount = best_action_info['pruning_amount']
        apply_quant = best_action_info['quantized']
        print(f"Applying Agent's best action: Pruning {pruning_amount*100:.0f}%, Quantization: {apply_quant}")
        try:
            agent_optimized_model = copy.deepcopy(baseline_model_cpu) # Fresh copy
            if pruning_amount > 0:
                agent_optimized_model = apply_l1_pruning(agent_optimized_model, pruning_amount, config)
            if apply_quant:
                agent_optimized_model = apply_dynamic_quantization(agent_optimized_model)

            results['Agent Optimized'] = evaluate_model(agent_optimized_model, test_loader, torch.device('cpu'), config)
        except Exception as e:
            print(f"Error applying or evaluating Agent's solution: {e}")
            results['Agent Optimized'] = {}
    else:
        print("Skipping Agent Optimized evaluation as best action could not be determined.")
        results['Agent Optimized'] = {} # Add empty entry

    # --- Format Results ---
    df_results = pd.DataFrame(results).T.fillna(0) # Fill NaNs from empty dicts with 0

    # Calculate derived metrics, checking if columns exist
    if 'flops' in df_results.columns:
        df_results['flops_g'] = (df_results['flops'] / 1e9).round(6)
    else: df_results['flops_g'] = 0

    if 'params' in df_results.columns:
        df_results['params_m'] = (df_results['params'] / 1e6).round(6)
    else: df_results['params_m'] = 0

    if 'energy_kwh' in df_results.columns:
        df_results['energy_mwh'] = (df_results['energy_kwh'] * 1000).round(6)
    else: df_results['energy_mwh'] = 0

    # Ensure co2 column exists, default to 0 if not
    if 'co2_eq_kg' not in df_results.columns:
         df_results['co2_eq_kg'] = 0


    # Select and order final columns
    final_columns = ['accuracy', 'params_m', 'flops_g', 'energy_mwh', 'co2_eq_kg']
    final_columns_present = [col for col in final_columns if col in df_results.columns]

    return df_results[final_columns_present]

def plot_results(df_results: pd.DataFrame, config: Config):
    """Generates and saves comparison plots."""
    print("\n--- Generating Visualizations ---")
    sns.set_style("whitegrid")

    # --- Bar Charts ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=20, weight='bold')

    metrics_to_plot = {
        ('accuracy', 'Accuracy Score', 'Greens_r', (0,0)),
        ('params_m', 'Parameters (M)', 'Blues_r', (0,1)),
        ('flops_g', 'GFLOPs', 'Oranges_r', (1,0)),
        ('energy_mwh', 'Energy (mWh)', 'Reds_r', (1,1))
    }

    for metric, ylabel, palette, pos in metrics_to_plot:
        ax = axes[pos[0], pos[1]]
        if metric in df_results.columns:
            sns.barplot(x=df_results.index, y=metric, data=df_results, ax=ax, palette=palette)
            ax.set_title(f'Model {ylabel.split(" ")[-1]} Comparison', fontsize=14, weight='bold')
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='x', rotation=15)
            # Add value labels
            for index, value in enumerate(df_results[metric]):
                 if pd.notna(value): # Check if value is not NaN before formatting
                      ax.text(index, value, f'{value:.4f}', ha='center', va='bottom')
        else:
             ax.set_title(f'{ylabel} Data Not Available', fontsize=14)
             ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, color='grey')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(config.COMPARISON_CHART_PATH)
    print(f"Comparison charts saved to: {config.COMPARISON_CHART_PATH}")
    # plt.show() # Optionally display plot

    # --- Pareto Front Scatter Plot ---
    if 'flops_g' in df_results.columns and 'accuracy' in df_results.columns:
        plt.figure(figsize=(12, 8))
        # Filter out rows where flops might be 0/NaN if needed, although plot should handle it
        plot_data = df_results[pd.to_numeric(df_results['flops_g'], errors='coerce').notnull() &
                               pd.to_numeric(df_results['accuracy'], errors='coerce').notnull()]

        sns.scatterplot(
            data=plot_data,
            x='flops_g',
            y='accuracy',
            hue=plot_data.index,
            s=200,
            style=plot_data.index,
            palette='viridis'
        )
        plt.title('Accuracy vs. Computational Cost Trade-off', fontsize=16, weight='bold')
        plt.xlabel('Computational Cost (GFLOPs)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        # Add text labels for each point
        for i in range(plot_data.shape[0]):
             plt.text(x=plot_data.flops_g[i]+0.0005, # Adjust offset slightly
                      y=plot_data.accuracy[i]+0.0005, # Adjust offset slightly
                      s=plot_data.index[i],
                      fontdict=dict(color='black',size=10))

        plt.legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(config.PARETO_CHART_PATH)
        print(f"Pareto front plot saved to: {config.PARETO_CHART_PATH}")
        # plt.show() # Optionally display plot
    else:
         print("Skipping Pareto plot: 'flops_g' or 'accuracy' data missing.")


if __name__ == "__main__":
    print("Executing Benchmark Script")
    config = Config()

    # Load data (only need test_loader)
    _, _, test_loader = load_data(config)

    if test_loader:
        # Load baseline model
        baseline_model = initialize_model(config)
        try:
            # Load trained weights onto CPU for benchmarking consistency
            baseline_model.load_state_dict(torch.load(config.BASELINE_MODEL_SAVE_PATH, map_location='cpu'))
            print(f"Baseline model weights loaded from {config.BASELINE_MODEL_SAVE_PATH}")

            # Run benchmark
            benchmark_results = run_full_benchmark(config, baseline_model, test_loader)

            print("\n\n--- FINAL BENCHMARK RESULTS ---")
            print(benchmark_results.to_string())

            # Save results to CSV
            os.makedirs(os.path.dirname(config.BENCHMARK_RESULTS_PATH), exist_ok=True)
            benchmark_results.to_csv(config.BENCHMARK_RESULTS_PATH)
            print(f"\nBenchmark results saved to: {config.BENCHMARK_RESULTS_PATH}")

            # Generate and save plots
            if not benchmark_results.empty:
                 plot_results(benchmark_results, config)


        except FileNotFoundError:
            print(f"Error: Baseline model weights not found at {config.BASELINE_MODEL_SAVE_PATH}")
            print("Please run src/train_baseline.py first.")
        except Exception as e:
             print(f"An unexpected error occurred during benchmarking: {e}")
    else:
        print("Failed to load data. Benchmark aborted.")