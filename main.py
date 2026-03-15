# from config import CONFIG
# from experiment_runner import run_experiment
# from experiments.model_list import MODEL_LIST


# for model_name in MODEL_LIST:

#     CONFIG["model_name"] = model_name

#     print(f"Running experiment for {model_name}")

#     run_experiment(CONFIG)


from config import CONFIG
from experiments.model_list import MODELS

from pipeline.data_pipeline import prepare_data_pipeline
from experiment_runner import run_experiment


def main():

    print("\nRunning data pipeline (once)...")

    train_loader, val_loader, test_loader, num_features, scaler, fmap = prepare_data_pipeline(CONFIG)

    print("\nStarting experiments...\n")

    for model in MODELS:

        run_experiment(
            model,
            train_loader,
            val_loader,
            test_loader,
            num_features
        )


if __name__ == "__main__":
    main()