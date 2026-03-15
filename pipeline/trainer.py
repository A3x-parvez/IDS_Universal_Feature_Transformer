# import os
# import time
# import torch
# import torch.nn as nn
# from datetime import datetime
# from tqdm.auto import tqdm

# from .evaluator import evaluate


# def train_model(model, train_loader, val_loader, config, model_dir):
#     print("Train function called ...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     print("\nTraining on device:", device)
#     if device.type == "cuda":
#         print("GPU:", torch.cuda.get_device_name(0))
#         print("CUDA Version:", torch.version.cuda)
#     else:
#         print("Using CPU")

#     opt = torch.optim.AdamW(
#         model.parameters(),
#         lr=config["learning_rate"],
#         weight_decay=config["weight_decay"]
#     )

#     criterion = nn.CrossEntropyLoss()
#     history = {"loss":[], "val_f1":[]}

#     # # ---------------------------------------
#     # # 📁 Create or reuse model folder
#     # # ---------------------------------------
#     # base_dir = os.path.join("results", "models")
#     # os.makedirs(base_dir, exist_ok=True)

#     # if config.get("resume", False):
#     #     model_dir = config["resume_path"]
#     # else:
#     #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     #     model_dir = os.path.join(base_dir, timestamp)

#     os.makedirs(model_dir, exist_ok=True)

#     # ---------------------------------------
#     # 🔁 Check for existing checkpoint
#     # ---------------------------------------
#     start_epoch = 0
#     checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_epoch")]

#     if checkpoints:
#         checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
#         latest_checkpoint = checkpoints[-1]
#         checkpoint_path = os.path.join(model_dir, latest_checkpoint)

#         print(f"\n🔄 Resuming from {latest_checkpoint}")

#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         opt.load_state_dict(checkpoint["optimizer_state_dict"])
#         start_epoch = checkpoint["epoch"]

#         print(f"Resumed at epoch {start_epoch}")

#     # ---------------------------------------
#     # 🚀 Training Loop
#     # ---------------------------------------
#     best_f1 = 0
#     start = time.time()

#     for epoch in range(start_epoch, config["epochs"]):
#         epoch_start = time.time()
#         model.train()
#         running_loss = 0

#         bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

#         for Xb, Mb, yb in bar:
#             Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)

#             opt.zero_grad()
#             out = model(Xb, Mb)
#             loss = criterion(out, yb)
#             loss.backward()
#             opt.step()

#             running_loss += loss.item()
#             bar.set_postfix(loss=loss.item())

#         avg_loss = running_loss / len(train_loader)
#         history["loss"].append(avg_loss)

#         val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, device)
#         history["val_f1"].append(val_f1)

#         print(f"\nLoss:{avg_loss:.4f}  ValAcc:{val_acc:.4f}  ValF1:{val_f1:.4f}")

#         # ✅ Save every epoch
#         epoch_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pt")

#         torch.save({
#             "epoch": epoch+1,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": opt.state_dict(),
#             "val_f1": val_f1,
#             "config": config
#         }, epoch_path)

#         print(f"Model saved: model_epoch_{epoch+1}.pt")
        
#         # check for the best model
#         if val_f1 > best_f1:
#             best_f1 = val_f1

#             best_path = os.path.join(model_dir, "best_model.pt")

#             torch.save(model.state_dict(), best_path)


#     print("\nTraining time:", round(time.time() - start, 2), "seconds")

#     return history


# new training

import os
import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

from .evaluator import evaluate


def train_model(model, train_loader, val_loader, config, model_dir):

    print("Train function called ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nTraining on device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        print("Using CPU")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # History for research + plotting
    # ---------------------------------------------------
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "epoch_time": []
    }

    os.makedirs(model_dir, exist_ok=True)

    # ---------------------------------------------------
    # Resume checkpoint
    # ---------------------------------------------------
    start_epoch = 0
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith("model_epoch")]

    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)

        print(f"\n🔄 Resuming from {latest_checkpoint}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

        print(f"Resumed at epoch {start_epoch}")

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    best_f1 = 0
    start = time.time()

    for epoch in range(start_epoch, config["epochs"]):

        epoch_start = time.time()

        model.train()

        running_loss = 0
        train_preds = []
        train_labels = []

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for Xb, Mb, yb in bar:

            Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)

            opt.zero_grad()

            out = model(Xb, Mb)

            loss = criterion(out, yb)

            loss.backward()

            opt.step()

            running_loss += loss.item()

            pred = torch.argmax(out, dim=1)

            train_preds.extend(pred.cpu().numpy())
            train_labels.extend(yb.cpu().numpy())

            bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)

        # ----------------------------
        # Train metrics
        # ----------------------------
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        # ----------------------------
        # Validation loss
        # ----------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for Xb, Mb, yb in val_loader:

                Xb, Mb, yb = Xb.to(device), Mb.to(device), yb.to(device)

                out = model(Xb, Mb)

                loss = criterion(out, yb)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        # ----------------------------
        # Validation metrics
        # ----------------------------
        val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # ----------------------------
        # Store history
        # ----------------------------
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(train_acc)
        history["train_f1"].append(train_f1)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        history["epoch_time"].append(epoch_time)

        print(
            f"\nTrainLoss:{avg_loss:.4f} "
            f"TrainAcc:{train_acc:.4f} "
            f"TrainF1:{train_f1:.4f} "
            f"ValLoss:{val_loss:.4f} "
            f"ValAcc:{val_acc:.4f} "
            f"ValF1:{val_f1:.4f} "
            f"EpochTime:{epoch_time:.2f}s"
        )

        # ---------------------------------------------------
        # Save epoch checkpoint
        # ---------------------------------------------------
        epoch_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pt")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_f1": val_f1,
            "config": config
        }, epoch_path)

        print(f"Model saved: model_epoch_{epoch+1}.pt")

        # ---------------------------------------------------
        # Best model
        # ---------------------------------------------------
        if val_f1 > best_f1:

            best_f1 = val_f1

            best_path = os.path.join(model_dir, "best_model.pt")

            torch.save(model.state_dict(), best_path)

            print("Best model updated")

    total_training_time = time.time() - start

    history["training_time"] = total_training_time

    print("\nTraining time:", round(total_training_time, 2), "seconds")

    return history

