import torch
import torch.nn as nn
import torch_pruning as tp
import copy
from src.config import Config
# from model import WeatherGRU # We might not need explicit type hint if not used internally

def apply_l1_pruning(model: nn.Module, amount: float, config: Config) -> nn.Module:
    """Applies L1 pruning, ignoring GRU layers. Returns model on CPU."""
    if amount <= 0:
        return copy.deepcopy(model).cpu()

    model_to_prune = copy.deepcopy(model).cpu()
    model_to_prune.eval()

    ignored_layers = [m for m in model_to_prune.modules() if isinstance(m, nn.GRU)]

    example_inputs = torch.randn(1, config.SEQUENCE_LENGTH, config.INPUT_DIM) # Batch size 1
    importance = tp.importance.MagnitudeImportance(p=1)

    try:
        # Suppress warnings from torch-pruning if needed
        import warnings
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", UserWarning) # Ignore unwrapped param warnings etc.
             pruner = tp.pruner.MagnitudePruner(
                 model_to_prune,
                 example_inputs=example_inputs,
                 importance=importance,
                 pruning_ratio=amount,
                 ignored_layers=ignored_layers,
                 # root_module_types=[nn.Linear], # Option: Only prune Linear layers
             )
             pruner.step()
    except Exception as e:
         print(f"Error during pruning step: {e}. Returning original model (CPU copy).")
         return copy.deepcopy(model).cpu()

    return model_to_prune

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Applies dynamic quantization to GRU and Linear layers. Returns model on CPU."""
    quantized_model = copy.deepcopy(model).cpu()
    quantized_model.eval()
    modules_to_quantize = {nn.GRU, nn.Linear}

    try:
        # Add a flag to check later if model is quantized
        quantized_model_dynamic = torch.quantization.quantize_dynamic(
            quantized_model, modules_to_quantize, dtype=torch.qint8
        )
        quantized_model_dynamic.quantized = True # Add flag
    except Exception as e:
        print(f"Error during dynamic quantization: {e}. Returning original model (CPU copy).")
        quantized_model_dynamic = copy.deepcopy(model).cpu()
        quantized_model_dynamic.quantized = False # Add flag


    return quantized_model_dynamic