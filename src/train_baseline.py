import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import os
from tqdm import tqdm
from src.config import Config
from src.data_processing import load_data
from src.model import initialize_model

def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader, config: Config) -> nn.Module:
    """
    Trains the baseline GRU model using specified configuration.

    Args:
        model (nn.Module): The model instance to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        config (Config): Configuration object with training parameters.

    Returns:
        nn.Module: The trained model with the best validation loss weights loaded.
    """
    criterion = nn.BCELoss() # Binary Cross Entropy for binary classification
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.SCHEDULER_T_MAX,
        eta_min=config.SCHEDULER_ETA_MIN
    )

    model.to(config.DEVICE)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    patience = 20 # Add simple early stopping based on validation loss

    print("--- Starting Baseline Model Training ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item()) # Show loss in progress bar

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")

        # Update learning rate scheduler
        scheduler.step()

        # Checkpoint saving and early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best validation loss: {best_val_loss:.4f}. Saving model state.")
            epochs_without_improvement = 0 # Reset counter
            # Save the best model immediately
            os.makedirs(os.path.dirname(config.BASELINE_MODEL_SAVE_PATH), exist_ok=True)
            torch.save(best_model_state, config.BASELINE_MODEL_SAVE_PATH)
        else:
             epochs_without_improvement += 1
             if epochs_without_improvement >= patience:
                 print(f"Early stopping triggered after {patience} epochs without improvement.")
                 break # Stop training

    # Load the best model state found during training
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"--- Best model state loaded (Val Loss: {best_val_loss:.4f}). ---")
    else:
        print("--- Warning: No best model state saved (training might have been too short or problematic). Using last state. ---")
        # Save the last state if no best state was found
        os.makedirs(os.path.dirname(config.BASELINE_MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), config.BASELINE_MODEL_SAVE_PATH)


    print("--- Baseline Model Training Finished ---")
    return model

if __name__ == "__main__":
    print("Executing Baseline Model Training Script")
    config = Config()

    # Load data
    train_loader, val_loader, _ = load_data(config) # We don't need test_loader here

    if train_loader and val_loader:
        # Initialize model
        baseline_model = initialize_model(config)

        # Train model
        trained_model = train_model(baseline_model, train_loader, val_loader, config)

        print(f"Baseline model trained and best state saved to: {config.BASELINE_MODEL_SAVE_PATH}")
    else:
        print("Failed to load data. Baseline model training aborted.")