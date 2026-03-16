CONFIG = {
    "data_files": [
        "data/raw/CIC-IDS-2017_fin_capped.csv",
        "data/raw/full_unsw.csv",
        "data/raw/NF_ToN_IoT_v3_clean.csv"
    ],
    "label_names": ["Label", "label", "attack_cat", "class"],

    "batch_size": 4096,
    "epochs": 20,
    "learning_rate": 0.0005,
    "weight_decay": 1e-4,

    "embedding_dim": 128,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.2,

    "seed": 42,
    "resume": True,
}
