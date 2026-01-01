import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(config):
    """
    Loads raw data, performs feature scaling, generates sequences, 
    and creates PyTorch tensors for training.
    
    Returns:
        dict: A dictionary containing train/val/test loaders and tensors.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(config.DATA_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {config.DATA_PATH}. Please verify the path.")

    # 2. Feature Engineering
    # Convert categorical 'weather' to binary target (1: Rain/Drizzle, 0: Others)
    df['target'] = df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)
    features = ['precipitation', 'temp_max', 'temp_min', 'wind']
    
    # 3. Normalization (Min-Max Scaling) to [0, 1]
    # Essential for neural network convergence
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    targets = df['target'].values
    
    # 4. Sequence Generation (Sliding Window)
    X, y = [], []
    for i in range(len(scaled_features) - config.SEQUENCE_LENGTH):
        X.append(scaled_features[i : i + config.SEQUENCE_LENGTH])
        y.append(targets[i + config.SEQUENCE_LENGTH])
    
    X = np.array(X)
    y = np.array(y)
    
    # 5. Stratified Data Splitting
    # Split Test Set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    # Split Validation Set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.SEED, stratify=y_temp
    )
    
    # 6. Tensor Conversion
    # Move data to GPU memory if available for faster training
    tensors = {
        'X_train': torch.FloatTensor(X_train),
        'y_train': torch.FloatTensor(y_train).unsqueeze(1),
        'X_val': torch.FloatTensor(X_val),
        'y_val': torch.FloatTensor(y_val).unsqueeze(1),
        'X_test': torch.FloatTensor(X_test),
        'y_test': torch.FloatTensor(y_test).unsqueeze(1)
    }
    
    print(f"Training Samples: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Save processed tensors for the RL Agent (Stage 2)
    torch.save(tensors, config.PROCESSED_DATA_SAVE_PATH)
    print(f"Processed data saved to {config.PROCESSED_DATA_SAVE_PATH}")
    
    return tensors