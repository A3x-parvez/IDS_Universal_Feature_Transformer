import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(model, loader, device):
    model.eval()
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for Xb, Mb, yb in loader:
            Xb, Mb = Xb.to(device), Mb.to(device)
            out = model(Xb, Mb)

            prob = torch.softmax(out,1)[:,1]
            pred = torch.argmax(out,1)

            preds.extend(pred.cpu().numpy())
            labels.extend(yb.numpy())
            probs.extend(prob.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    return acc, f1, prec, rec, labels, probs
