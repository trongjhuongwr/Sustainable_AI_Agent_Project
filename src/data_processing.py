import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.config import Config

def create_sequences(features: np.ndarray, target: np.ndarray, seq_length: int):
    """Creates sequences of data for time series forecasting."""
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_data(config: Config):
    """Loads, preprocesses, splits data and returns DataLoaders."""
    try:
        df = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.DATA_PATH}")
        return None, None, None

    df['weather_numeric'] = df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)
    df = df.drop(columns=['date', 'weather'])

    features_df = df.drop('weather_numeric', axis=1)
    target_series = df['weather_numeric']

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)

    X, y = create_sequences(scaled_features, target_series.values, config.SEQUENCE_LENGTH)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    relative_val_size = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=config.RANDOM_STATE, stratify=y_temp
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True if config.DEVICE != torch.device('cpu') else False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True if config.DEVICE != torch.device('cpu') else False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True if config.DEVICE != torch.device('cpu') else False)

    print(f"Data processed: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    test_dist = pd.Series(y_test).value_counts(normalize=True)
    print(f"Class distribution in test set:\n{test_dist}")

    return train_loader, val_loader, test_loader