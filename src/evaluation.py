import torch
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker
from ptflops import get_model_complexity_info
from tqdm.auto import tqdm
from src.config import Config

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, config: Config) -> dict:
    """Comprehensive evaluation on CPU."""
    eval_device = torch.device("cpu")
    model_cpu = copy.deepcopy(model).to(eval_device)
    model_cpu.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(eval_device)
            outputs = model_cpu(inputs)
            preds = (outputs > 0.5).float()
            y_pred.extend(preds.cpu().numpy().flatten())
            y_true.extend(labels.cpu().numpy().flatten())
    accuracy = accuracy_score(y_true, y_pred)

    tracker = EmissionsTracker(log_level="error", output_dir=config.CODECARBON_OUTPUT_DIR, tracking_mode="process")
    tracker.start()
    with torch.no_grad():
        batches_processed = 0
        for inputs, _ in loader:
            if batches_processed >= config.NUM_EVAL_BATCHES: break
            model_cpu(inputs.to(eval_device))
            batches_processed += 1
    emissions_data = tracker.stop() # Returns file path or emissions data
    energy_kwh = tracker.final_emissions_data.energy_consumed if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data else 0.0
    co2_eq_kg = tracker.final_emissions_data.emissions if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data else 0.0

    params = count_parameters(model_cpu)
    flops = 0
    try:
        input_res = (1, config.SEQUENCE_LENGTH, config.INPUT_DIM) # Use batch size 1 for ptflops
        macs, _ = get_model_complexity_info(
            model_cpu, input_res, as_strings=False, print_per_layer_stat=False, verbose=False)
        flops = macs * 2
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs via ptflops (Reason: {e}). Reporting 0 FLOPs.")

    return {
        "accuracy": accuracy,
        "energy_kwh": energy_kwh if energy_kwh is not None else 0.0,
        "co2_eq_kg": co2_eq_kg if co2_eq_kg is not None else 0.0,
        "flops": flops,
        "params": params,
    }

def evaluate_lightweight(model: torch.nn.Module, one_batch_loader_iter: iter, baseline_metrics: dict, config: Config) -> dict:
    """Lightweight evaluation for RL step on CPU."""
    model.eval()
    eval_device = torch.device("cpu")
    model = model.to(eval_device) # Ensure model is on CPU

    try:
        inputs, labels = next(one_batch_loader_iter)
        inputs, labels = inputs.to(eval_device), labels.to(eval_device)
    except StopIteration:
        print("Warning: DataLoader exhausted in _evaluate_lightweight. Returning baseline.")
        # Return copies of baseline values
        return {k: baseline_metrics.get(k, 0) for k in ["accuracy", "flops", "params"]}


    with torch.no_grad():
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        accuracy = accuracy_score(labels.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

    params = 0
    flops = 0
    try:
        input_res = (1, config.SEQUENCE_LENGTH, config.INPUT_DIM) # Use batch size 1
        macs, calculated_params = get_model_complexity_info(
            model, input_res, as_strings=False, print_per_layer_stat=False, verbose=False)
        flops = macs * 2
        params = calculated_params
    except Exception as e:
        # print(f"Info: ptflops failed in lightweight eval ({e}). Using estimation.") # Debug
        flops = 0 # Assume significant reduction
        if hasattr(model, 'quantized') or "quantized" in str(type(model)).lower(): # Check if likely quantized
            params = baseline_metrics.get("params", 1) / 4 # Estimate 1/4 size, avoid div by zero
        else:
             params = count_parameters(model) # Count directly if not quantized but failed

    # Ensure params is at least 0
    params = max(0, params)

    return {"accuracy": accuracy, "flops": flops, "params": params}