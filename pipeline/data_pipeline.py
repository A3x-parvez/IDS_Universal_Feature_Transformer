import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from pipeline.utils import clean_column_names, normalize_feature_names, FEATURE_SYNONYMS
# run the pipeline one by one


# Load datasets step 1
def load_datasets(files):
    print("Load datsets ...")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"Loaded {f} → shape {df.shape}")
        dfs.append(df)
    return dfs


# Unify labels step 2
def unify_labels(dfs, label_names):
    print("Unify the label names ..")
    for df in dfs:
        for col in label_names:
            if col in df.columns:
                df.rename(columns={col: "target"}, inplace=True)

        if "target" not in df.columns:
            raise ValueError("Target column not found!")

    print("Labels unified.")
    return dfs


# Build feature map
def build_feature_map(dfs):
    print("Build Feature Map ...")
    features = set()
    for df in dfs:
        features.update(df.columns)

    features.remove("target")
    features = sorted(list(features))

    fmap = {f:i for i,f in enumerate(features)}

    print("Total universal features:", len(fmap))
    return fmap


# convert to universal feature map
def convert_to_universal(df, fmap):
    print("Convert to universal feature is started ..")
    X = np.zeros((len(df), len(fmap)))
    mask = np.zeros_like(X)

    for col in df.columns:
        if col == "target":
            continue
        if col in fmap:
            idx = fmap[col]
            vals = df[col].fillna(0).values
            X[:, idx] = vals
            mask[:, idx] = 1

    y = df["target"].values
    return X, mask, y


# Preapare Data
def prepare_data(dfs, fmap):
    print("Prepare data is working ..")
    Xs, Ms, Ys = [], [], []

    for df in dfs:
        X, M, y = convert_to_universal(df, fmap)
        Xs.append(X)
        Ms.append(M)
        Ys.append(y)

    X = np.vstack(Xs)
    M = np.vstack(Ms)
    y = np.hstack(Ys)

    print("Final shape:", X.shape)
    return X, M, y


def normalize_features(X):
    print("Scale the Features")
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


class TabularDataset(Dataset):
    def __init__(self, X, M, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx]



def create_loaders(X, M, y, batch_size, seed=42):
    print("Create loader started ...")
    dataset = TabularDataset(X, M, y)

    total = len(dataset)
    train_len = int(0.7 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator
    )

    num_workers = min(8, os.cpu_count())
    print(f"Number of workers use to load data {num_workers}")

    print(f"Train:{train_len}  Val:{val_len}  Test:{test_len}")
    print("Create loader Ended.")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),

        DataLoader(val_set, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),

        DataLoader(test_set, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True)
    )



def prepare_data_pipeline(config):

    print("\n=========== DATA PIPELINE STARTED ===========")

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    fmap_path = os.path.join(processed_dir, "feature_map.json")
    scaler_path = os.path.join(processed_dir, "scaler.save")

    # ------------------------------------------------
    # Load datasets
    # ------------------------------------------------
    dfs = load_datasets(config["data_files"])

    # ------------------------------------------------
    # Clean column names
    # ------------------------------------------------
    dfs = [clean_column_names(df) for df in dfs]

    # ------------------------------------------------
    # Unify labels
    # ------------------------------------------------
    dfs = unify_labels(dfs, config["label_names"])

    # ------------------------------------------------
    # Normalize feature synonyms
    # ------------------------------------------------
    dfs = normalize_feature_names(dfs, FEATURE_SYNONYMS)

    # ------------------------------------------------
    # Build universal feature map
    # ------------------------------------------------
    fmap = build_feature_map(dfs)

    with open(fmap_path, "w") as f:
        json.dump(fmap, f)

    print("Feature map saved.")

    # ------------------------------------------------
    # Convert to universal format
    # ------------------------------------------------
    X, M, y = prepare_data(dfs, fmap)

    # ------------------------------------------------
    # Normalize features
    # ------------------------------------------------
    X, scaler = normalize_features(X)

    joblib.dump(scaler, scaler_path)

    print("Scaler saved.")

    # ------------------------------------------------
    # Create loaders
    # ------------------------------------------------
    train_loader, val_loader, test_loader = create_loaders(
        X,
        M,
        y,
        config["batch_size"],
        config["seed"]
    )

    num_features = X.shape[1]

    print("=========== DATA PIPELINE FINISHED ===========\n")

    return train_loader, val_loader, test_loader, num_features, scaler, fmap