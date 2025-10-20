import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import copy
import warnings
import itertools # For resetting dataloader iterator cycle

# Import relative to Kaggle working dir
from config import Config
# from model import WeatherGRU # Type hint if needed
from evaluation import evaluate_lightweight
from optimization import apply_l1_pruning, apply_dynamic_quantization

class SustainableAIAgentEnv(gym.Env):
    """Custom Gymnasium environment for RL-based model optimization."""
    metadata = {'render_modes': []}

    def __init__(self, baseline_model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, config: Config):
        super(SustainableAIAgentEnv, self).__init__()
        self.config = config
        self.baseline_model = copy.deepcopy(baseline_model).cpu()
        self.eval_loader = eval_loader
        # Use itertools.cycle for infinite iterator
        self._eval_loader_iter = itertools.cycle(self.eval_loader)

        print("Calculating baseline metrics for RL environment (using lightweight eval)...")
        # Check loader has data
        try:
             first_batch = next(self._eval_loader_iter)
        except StopIteration:
             raise ValueError("Evaluation DataLoader is empty. Cannot initialize environment.")

        # Calculate baseline metrics for reward reference
        # Pass a copy of the first batch iterator to avoid consuming from the main one
        temp_iter = iter([first_batch])
        self.baseline_metrics = evaluate_lightweight(
            self.baseline_model, temp_iter, {}, self.config # Pass empty dict initially
        )

        print(f"Baseline Metrics (Lightweight Eval): {self.baseline_metrics}")
        if self.baseline_metrics.get("flops", 0) <= 0 or self.baseline_metrics.get("params", 0) <= 0:
            warnings.warn("Baseline FLOPs or Params are zero/negative according to lightweight eval. Resource rewards might be incorrect.")

        # Action Space
        self.pruning_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        num_pruning_levels = len(self.pruning_levels)
        self.action_space = spaces.Discrete(num_pruning_levels * 2)
        print(f"Action Space: {self.action_space}")

        # Observation Space
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,), dtype=np.float32
        )
        print(f"Observation Space: {self.observation_space}")

    def _get_one_batch_iter(self):
         """Gets the next batch using the cycling iterator."""
         return iter([next(self._eval_loader_iter)])

    def step(self, action: int):
        terminated = True
        truncated = False
        try:
            num_pruning_levels = len(self.pruning_levels)
            pruning_index = action % num_pruning_levels
            pruning_amount = self.pruning_levels[pruning_index]
            apply_quant = (action >= num_pruning_levels)

            # --- Apply Optimizations ---
            # Start fresh from baseline CPU model each step
            current_model = copy.deepcopy(self.baseline_model).cpu()
            if pruning_amount > 0:
                current_model = apply_l1_pruning(current_model, pruning_amount, self.config)
            if apply_quant:
                current_model = apply_dynamic_quantization(current_model)
            optimized_model = current_model
            # ---------------------------

            metrics = evaluate_lightweight(
                optimized_model, self._get_one_batch_iter(), self.baseline_metrics, self.config
            )

            # --- Calculate Reward ---
            accuracy_delta = metrics["accuracy"] - self.baseline_metrics.get("accuracy", 0.0)
            accuracy_threshold_relative = self.baseline_metrics.get("accuracy", 0.0) * 0.95
            accuracy_reward = -5.0 if metrics["accuracy"] < accuracy_threshold_relative else accuracy_delta * 10.0

            baseline_flops = self.baseline_metrics.get("flops", 1.0) # Avoid div by zero
            flops_reduction = 1.0 - (metrics.get("flops", 0.0) / baseline_flops) if baseline_flops > 0 else 0.0

            baseline_params = self.baseline_metrics.get("params", 1.0) # Avoid div by zero
            params_reduction = 1.0 - (metrics.get("params", 0.0) / baseline_params) if baseline_params > 0 else 0.0

            flops_reduction = np.clip(flops_reduction, 0.0, 1.0)
            params_reduction = np.clip(params_reduction, 0.0, 1.0)

            resource_reward = (flops_reduction * 1.5) + (params_reduction * 0.5)
            # inaction_penalty = -0.1 if pruning_amount == 0 and not apply_quant else 0.0 # Optional
            reward = accuracy_reward + resource_reward # + inaction_penalty
            # ------------------------

            # --- Construct Observation ---
            obs = np.array([
                np.clip(metrics.get("accuracy", 0.0), 0.0, 1.0),
                np.clip(accuracy_delta, -1.0, 1.0),
                params_reduction,
                flops_reduction
            ], dtype=np.float32)
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
            # ---------------------------

            info = {
                "pruning_amount": pruning_amount, "quantized": apply_quant,
                "accuracy": metrics.get("accuracy", 0.0), "flops_reduction": flops_reduction,
                "params_reduction": params_reduction, "reward": reward,
                "raw_flops": metrics.get("flops", 0.0), "raw_params": metrics.get("params", 0.0)
            }
            return obs, reward, terminated, truncated, info

        except Exception as e:
            print(f"Error during environment step: {e}. Applying large penalty.")
            obs = np.array([self.baseline_metrics.get("accuracy", 0.0), 0.0, 0.0, 0.0], dtype=np.float32)
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
            return obs, -10.0, True, False, {"error": str(e)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_obs = np.array([
            self.baseline_metrics.get("accuracy", 0.0), 0.0, 0.0, 0.0
        ], dtype=np.float32)
        initial_obs = np.clip(initial_obs, self.observation_space.low, self.observation_space.high)
        info = {}
        # No need to reset iterator here as itertools.cycle handles it
        return initial_obs, info

    def close(self):
        pass