import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.config import Config
from src.environment import SustainableAIAgentEnv
from src.model import initialize_model
from src.data_processing import load_data

# Global list to store history across chunks (if script runs continuously)
# Note: If script is stopped and restarted, this history is lost unless saved to file.
experiment_history = []

class HistoryCallback(BaseCallback):
    """
    A custom callback that collects info dictionary from each environment step.
    """
    def __init__(self, verbose=0):
        super(HistoryCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # `info` contains the dictionary returned by env.step
        # In VecEnvs, 'infos' is a list of dicts, one for each env.
        infos = self.locals.get("infos", [])
        for info in infos:
            # Check if it's the actual end of an episode (not just truncation)
            # And if the info dict contains our custom keys
            if info.get("terminal_observation") is not None and "pruning_amount" in info:
                # Store a copy of the relevant info
                log_entry = {k: info[k] for k in ["pruning_amount", "quantized", "accuracy",
                                                  "flops_reduction", "params_reduction", "reward"] if k in info}
                if log_entry: # Ensure it's not empty
                    experiment_history.append(log_entry)
        return True # Continue training

def train_ppo_agent(config: Config, baseline_model: nn.Module, train_loader, test_loader):
    """
    Initializes or loads a PPO agent and trains it on the SustainableAIAgentEnv.

    Args:
        config (Config): Configuration object.
        baseline_model (nn.Module): The pre-trained baseline model.
        train_loader: DataLoader used by the environment (e.g., test_loader for eval).
    """
    print("--- Initializing RL Environment ---")
    # Use test_loader for environment evaluation as it's typically smaller/faster per batch
    # Make a dummy VecEnv wrapper (required by SB3) even for a single env
    # Wrap with Monitor for SB3 logging compatibility (rewards, episode lengths)
    env = make_vec_env(lambda: Monitor(SustainableAIAgentEnv(baseline_model, test_loader, config)), n_envs=1)


    # Initialize or load Agent
    agent = None
    if os.path.exists(config.AGENT_SAVE_PATH):
        try:
            print(f"--- Loading existing Agent from: {config.AGENT_SAVE_PATH} ---")
            agent = PPO.load(config.AGENT_SAVE_PATH, env=env)
            print("Agent loaded successfully.")
        except Exception as e:
            print(f"Error loading agent: {e}. Creating a new agent.")
            os.remove(config.AGENT_SAVE_PATH) # Remove potentially corrupted file
            agent = None

    if agent is None:
        print("--- Creating a new PPO Agent ---")
        agent = PPO(
            "MlpPolicy", # Standard Multi-Layer Perceptron policy for Box/Discrete spaces
            env,
            verbose=1, # Print training progress
            device=config.DEVICE, #'auto' might work better sometimes
            tensorboard_log=os.path.join(config.CODECARBON_OUTPUT_DIR, "ppo_tensorboard/") # Log to results
            # Add other PPO hyperparameters here if needed (e.g., n_steps, batch_size, learning_rate)
            # n_steps=2048, # Default, steps collected per update
            # batch_size=64, # Default minibatch size
            # learning_rate=3e-4 # Default learning rate
        )
        print("New Agent created successfully.")

    # Training loop in chunks
    timesteps_trained_so_far = agent.num_timesteps
    if timesteps_trained_so_far >= config.TOTAL_TIMESTEPS:
        print(f"\n--- Agent already trained for {timesteps_trained_so_far} steps. Skipping training. ---")
    else:
        remaining_timesteps = config.TOTAL_TIMESTEPS - timesteps_trained_so_far
        chunks_to_run = int(np.ceil(remaining_timesteps / config.TIMESTEPS_PER_CHUNK))

        print(f"\n--- Starting/Resuming Agent Training ---")
        print(f"Target: {config.TOTAL_TIMESTEPS} steps | Already Trained: {timesteps_trained_so_far} steps.")
        print(f"Will run {chunks_to_run} more chunk(s).")

        callback = HistoryCallback()

        for i in range(chunks_to_run):
            current_chunk = i + 1
            print(f"\n--- Running Chunk {current_chunk}/{chunks_to_run} ---")

            # Calculate timesteps for this chunk, ensuring not to exceed total
            timesteps_this_chunk = min(config.TIMESTEPS_PER_CHUNK, config.TOTAL_TIMESTEPS - agent.num_timesteps)
            if timesteps_this_chunk <= 0: break # Should not happen with ceil logic, but safety check


            try:
                agent.learn(
                    total_timesteps=timesteps_this_chunk,
                    reset_num_timesteps=False, # Continue counting total steps
                    progress_bar=True,
                    callback=callback # Use our custom callback
                )
            except Exception as e:
                 print(f"\n--- Error during agent.learn(): {e} ---")
                 print("Saving agent before exiting.")
                 break # Stop training on error


            # Save agent after each chunk
            agent.save(config.AGENT_SAVE_PATH)
            current_total_steps = agent.num_timesteps
            print(f"Agent saved. Total steps trained: {current_total_steps}.")

            # Check if target reached
            if current_total_steps >= config.TOTAL_TIMESTEPS:
                print("\n--- Target number of training steps reached. ---")
                break

    print("\n--- Agent Training Finished ---")

    # --- Analyze and display results from history ---
    if experiment_history:
        history_df = pd.DataFrame(experiment_history)
        print("\n--- History of Solutions Explored by Agent (Last Session): ---")
        print(history_df.to_string())

        if not history_df.empty:
            # Find the best solution based on the highest reward in the recorded history
            best_solution_row = history_df.loc[history_df['reward'].idxmax()]
            print("\n" + "="*50)
            print("BEST SOLUTION FOUND (in last session's history) 🏆")
            print("="*50)
            print(best_solution_row)
            print("="*50)
        else:
            print("\nHistory DataFrame is empty. Cannot determine best solution from history.")

    else:
        print("\nNo experiment history recorded in this session. Cannot display best solution from history.")
        print("Will rely on agent.predict() during benchmarking.")

if __name__ == "__main__":
    print("Executing Agent Training Script")
    config = Config()

    # Load data (only need test_loader for the env here)
    _, _, test_loader = load_data(config)

    if test_loader:
        # Load baseline model
        baseline_model = initialize_model(config)
        try:
            # Load pre-trained weights, ensuring they are on the correct device initially
            baseline_model.load_state_dict(torch.load(config.BASELINE_MODEL_SAVE_PATH, map_location=config.DEVICE))
            baseline_model.to(config.DEVICE) # Ensure model is on the configured device
            print(f"Baseline model weights loaded from {config.BASELINE_MODEL_SAVE_PATH}")

            # Train the agent
            train_ppo_agent(config, baseline_model, test_loader, test_loader) # Pass test_loader twice for env

        except FileNotFoundError:
            print(f"Error: Baseline model weights not found at {config.BASELINE_MODEL_SAVE_PATH}")
            print("Please run src/train_baseline.py first.")
        except Exception as e:
             print(f"An unexpected error occurred: {e}")
    else:
        print("Failed to load data. Agent training aborted.")