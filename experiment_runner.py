import os
import json
import torch

from config import CONFIG
from pipeline.utils import set_seed
from pipeline.trainer import train_model
from pipeline.evaluator import evaluate
from pipeline.plotting import generate_all_plots
from pipeline.utils import create_experiment_run, update_experiment_summary

from models import get_model


def run_experiment(model_name, train_loader, val_loader, test_loader, num_features):

    # ---------------------------------------
    # Set random seed for reproducibility
    # ---------------------------------------
    set_seed(CONFIG["seed"])

    # -------------------------------------------------
    # Create experiment run folder
    # -------------------------------------------------
    run_dir, graphs_dir, model_dir = create_experiment_run(model_name)

    print("\nRunning experiment:", model_name)
    print("Run directory:", run_dir)

    # -------------------------------------------------
    # Save config
    # -------------------------------------------------
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    # -------------------------------------------------
    # Build Model
    # -------------------------------------------------
    model = get_model(model_name, num_features, CONFIG)

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    history = train_model(
        model,
        train_loader,
        val_loader,
        CONFIG,
        model_dir
    )

    # -------------------------------------------------
    # Test evaluation
    # -------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acc, f1, prec, rec, labels, probs = evaluate(model, test_loader, device)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    metrics = {
        "test_accuracy": acc,
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "training_time": history["training_time"]
    }

    # -------------------------------------------------
    # Save metrics
    # -------------------------------------------------
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # -------------------------------------------------
    # Save training history
    # -------------------------------------------------
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # -------------------------------------------------
    # Generate plots
    # -------------------------------------------------
    generate_all_plots(history, labels, probs, graphs_dir)

    # -------------------------------------------------
    # Update global experiment CSV
    # -------------------------------------------------
    run_name = os.path.basename(run_dir)

    update_experiment_summary(
        model_name,
        run_name,
        history,
        metrics
    )

    print("\nExperiment finished:", model_name)