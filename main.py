import argparse
import copy
from pathlib import Path

import torch

from swigat.data import build_loaders
from swigat.engine import evaluate, fit, summarize_runs
from swigat.model import SwiGAT
from swigat.utils import load_config, resolve_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate SwiGAT.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", help="Checkpoint used in test mode.")
    parser.add_argument("--device", help="Override config device, e.g. cuda:0 or cpu.")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    parser.add_argument("--batch-size", type=int, help="Override the batch size.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one synthetic forward pass.")
    return parser.parse_args()


def build_model(config, pretrained=None):
    model_cfg = config["model"]
    return SwiGAT(
        num_classes=config["data"]["num_classes"],
        feature_dim=model_cfg["feature_dim"],
        ffb_hidden_dim=model_cfg["ffb_hidden_dim"],
        gaze_hidden_dim=model_cfg["gaze_hidden_dim"],
        gat_heads=model_cfg["gat_heads"],
        dropout=model_cfg["dropout"],
        pretrained_backbone=model_cfg["pretrained_backbone"] if pretrained is None else pretrained,
    )


def smoke_test(config, device):
    model = build_model(config, pretrained=False).to(device).eval()
    size = config["data"]["image_size"]
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, size, size, device=device), torch.rand(1, 1, size, size, device=device))
    expected = (1, config["data"]["num_classes"])
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)} != {expected}")
    print(f"Smoke test passed: logits shape = {tuple(logits.shape)}")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    device = resolve_device(config.get("device", "auto"))

    if args.smoke_test:
        smoke_test(config, device)
        return

    train_loader, test_loader = build_loaders(config)
    if args.mode == "test":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in test mode")
        model = build_model(config, pretrained=False).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        metrics = evaluate(model, test_loader, device, config["data"]["num_classes"])
        print(metrics)
        return

    results = []
    for seed in config.get("seeds", [config.get("seed", 42)]):
        run_config = copy.deepcopy(config)
        run_config["seed"] = seed
        set_seed(seed)
        model = build_model(run_config).to(device)
        run_dir = Path(run_config["output_dir"]) / f"seed_{seed}"
        results.append(fit(model, train_loader, test_loader, run_config, device, run_dir))
    summary = summarize_runs(results)
    save_json({"runs": results, "summary": summary}, Path(config["output_dir"]) / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
