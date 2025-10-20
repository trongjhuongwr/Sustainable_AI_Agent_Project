import torch
import os

class Config:
    """Stores configuration parameters for the project."""
    # --- Paths ---
    # Assume data is in /kaggle/input/...
    # For local execution, you might change DATA_PATH to '../data/seattle-weather.csv'
    DATA_PATH = '/kaggle/input/weather-prediction/seattle-weather.csv'

    # Output directory within Kaggle's working space
    OUTPUT_DIR = "/kaggle/working/results" # Changed for Kaggle

    BASELINE_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "baseline_gru_model.pth")
    AGENT_SAVE_PATH = os.path.join(OUTPUT_DIR, "sustainable_ai_agent_ppo.zip")
    BENCHMARK_RESULTS_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    COMPARISON_CHART_PATH = os.path.join(OUTPUT_DIR, "comparison_charts.png")
    PARETO_CHART_PATH = os.path.join(OUTPUT_DIR, "pareto_front.png")
    CODECARBON_OUTPUT_DIR = OUTPUT_DIR # CodeCarbon logs go here
    TENSORBOARD_LOG_DIR = os.path.join(OUTPUT_DIR, "ppo_tensorboard/")

    # --- Data Parameters ---
    SEQUENCE_LENGTH = 30
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1 # Relative to (1 - TEST_SIZE)
    RANDOM_STATE = 42

    # --- GRU Model Parameters ---
    INPUT_DIM = 4
    HIDDEN_DIM = 256
    N_LAYERS = 2
    OUTPUT_DIM = 1
    DROPOUT_RATE = 0.2

    # --- Baseline Model Training Parameters ---
    BATCH_SIZE = 64
    EPOCHS = 500
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    SCHEDULER_T_MAX = 50
    SCHEDULER_ETA_MIN = 1e-6
    EARLY_STOPPING_PATIENCE = 20 # Added patience

    # --- RL Agent (PPO) Parameters ---
    TOTAL_TIMESTEPS = 10000
    TIMESTEPS_PER_CHUNK = 2000

    # --- Device Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Evaluation Parameters ---
    NUM_EVAL_BATCHES = 10 # Batches for energy estimation

# --- Create output directory on Kaggle ---
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"Configuration loaded. Output directory: {Config.OUTPUT_DIR}")
print(f"Running on device: {Config.DEVICE}")