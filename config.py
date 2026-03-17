CONFIG = {
    "data_files": [
        "data/raw/CIC-IDS-2017_fin_capped.csv",
        "data/raw/full_unsw.csv",
        "data/raw/NF_ToN_IoT_v3_clean.csv"
    ],
    "label_names": ["Label", "label", "attack_cat", "class"],

    "batch_size": 2048,
    "epochs": 20,
    "learning_rate": 0.0001,
    "weight_decay": 1e-4,

    "embedding_dim": 256,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.2,

    "seed": 42,
    "resume": True,
}
