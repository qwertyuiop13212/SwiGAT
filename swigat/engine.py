from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .metrics import classification_metrics, weighted_prf
from .utils import AverageMeter


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    for images, gazes, labels in tqdm(loader, desc="train", leave=False):
        images, gazes, labels = images.to(device), gazes.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, gazes)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), labels.size(0))
    return losses.average


@torch.no_grad()
def evaluate(model, loader, device, num_classes, use_amp=False):
    model.eval()
    labels_all, probabilities_all, batch_prf = [], [], []
    for images, gazes, labels in tqdm(loader, desc="evaluate", leave=False):
        images, gazes = images.to(device), gazes.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, gazes)
        batch_probabilities = logits.softmax(dim=1).cpu().numpy()
        batch_labels = labels.numpy()
        labels_all.extend(batch_labels.tolist())
        probabilities_all.extend(batch_probabilities.tolist())
        batch_prf.append(weighted_prf(batch_labels, batch_probabilities))
    metrics = classification_metrics(labels_all, probabilities_all, num_classes)
    metrics["precision"] = float(np.mean([value[0] for value in batch_prf]))
    metrics["recall"] = float(np.mean([value[1] for value in batch_prf]))
    metrics["f1"] = float(np.mean([value[2] for value in batch_prf]))
    return metrics


def fit(model, train_loader, test_loader, config, device, run_dir):
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_metrics = -1.0, None
    for epoch in range(1, int(training["epochs"]) + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        metrics = evaluate(model, test_loader, device, config["data"]["num_classes"], use_amp)
        print(f"Epoch {epoch:03d} | loss={loss:.4f} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if metrics["f1"] > best_f1:
            best_f1, best_metrics = metrics["f1"], metrics
            torch.save(
                {"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": metrics},
                run_dir / "best.pt",
            )
    return best_metrics


def summarize_runs(results):
    keys = results[0].keys()
    return {
        key: {"mean": float(np.nanmean([run[key] for run in results])), "std": float(np.nanstd([run[key] for run in results]))}
        for key in keys
    }

