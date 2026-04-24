from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hrtf_personalization.config import load_yaml
from hrtf_personalization.data import (
    CIPICPreparedDataset,
    collate_prepared_samples,
    fetch_hrtfcnn_assets,
    prepare_from_hrtfcnn_repo,
)
from hrtf_personalization.evaluation import log_spectral_distance, rmse
from hrtf_personalization.models import BaselineHRTFEstimator, ConditionalHRTFEstimator
from hrtf_personalization.measurement import collect_anthropometrics, load_anthropometrics_json
from hrtf_personalization.prediction import PredictionInputs, predict_sofa_from_image
from hrtf_personalization.preprocessing import EarImagePreprocessor
from hrtf_personalization.sofa import export_simple_free_field_hrir
from hrtf_personalization.training import Trainer, TrainingConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ANTHRO_PATH = _PROJECT_ROOT / "configs" / "default-anthro.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrtf-personalization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_assets = subparsers.add_parser("fetch-hrtfcnn-assets")
    fetch_assets.add_argument("--config", required=True)
    fetch_assets.add_argument("--overwrite", action="store_true")

    prepare = subparsers.add_parser("prepare-cipic")
    prepare.add_argument("--config", required=True)

    train = subparsers.add_parser("train-baseline")
    train.add_argument("--config", required=True)

    train_conditional = subparsers.add_parser("train-conditional")
    train_conditional.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)

    export = subparsers.add_parser("export-sofa")
    export.add_argument("--config", required=True)

    measure = subparsers.add_parser("measure-anthro")
    measure.add_argument("--front-image", required=True)
    measure.add_argument("--side-image", required=True)
    measure.add_argument("--output", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image")
    predict.add_argument("--left-image")
    predict.add_argument("--right-image")
    predict.add_argument("--image-ear", choices=["auto", "left", "right"], default="auto")
    predict.add_argument("--anthro", help="Comma-separated anthropometric values.")
    predict.add_argument("--anthro-json")
    predict.add_argument("--front-image")
    predict.add_argument("--side-image")
    predict.add_argument("--measure-output")
    predict.add_argument("--template-sofa", required=True)
    predict.add_argument("--output-sofa")
    predict.add_argument("--device", default="cpu")

    return parser


def cmd_fetch_hrtfcnn_assets(config_path: str, overwrite: bool) -> int:
    config = load_yaml(config_path)
    dataset_config = config["dataset"]
    if dataset_config["source_layout"] != "hrtfcnn":
        raise SystemExit(f"Unsupported source_layout: {dataset_config['source_layout']}")
    fetch_hrtfcnn_assets(dataset_config["hrtfcnn_repo_root"], overwrite=overwrite)
    print(f"Fetched HRTFCNN assets under {dataset_config['hrtfcnn_repo_root']}")
    return 0


def cmd_prepare_cipic(config_path: str) -> int:
    config = load_yaml(config_path)
    dataset_config = config["dataset"]
    if dataset_config["source_layout"] != "hrtfcnn":
        raise SystemExit(f"Unsupported source_layout: {dataset_config['source_layout']}")

    ear_preprocessor = EarImagePreprocessor(
        image_size=dataset_config["image_size"],
        crop_x=dataset_config["crop"]["x"],
        crop_y=dataset_config["crop"]["y"],
        crop_width=dataset_config["crop"]["width"],
        crop_height=dataset_config["crop"]["height"],
        blur_kernel=dataset_config["blur_kernel"],
        low_threshold=dataset_config["edge_detection"]["low_threshold"],
        high_threshold=dataset_config["edge_detection"]["high_threshold"],
    )
    summaries = prepare_from_hrtfcnn_repo(
        repo_root=dataset_config["hrtfcnn_repo_root"],
        prepared_root=dataset_config["prepared_root"],
        anthropometric_dim=dataset_config["anthropometrics"]["use_first_n"],
        ear_preprocessor=ear_preprocessor,
    )
    prepared_root = Path(dataset_config["prepared_root"])
    sample_count = sum(summary.num_directions * 2 for summary in summaries)
    print(f"Prepared {sample_count} samples from {len(summaries)} subjects at {prepared_root}")
    return 0


def cmd_train_baseline(config_path: str) -> int:
    config = load_yaml(config_path)
    _, dataloader = _build_dataset_and_loader(config)
    model = BaselineHRTFEstimator(
        anthropometric_dim=config["model"]["anthropometric_dim"],
        hrtf_dim=config["model"]["hrtf_dim"],
        ear_image_size=config["model"]["ear_image_size"],
        ear_side_dim=config["model"].get("ear_side_dim", 1),
    )
    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            device=config["training"]["device"],
            use_direction=False,
            optimizer=config["training"].get("optimizer", "adam"),
            loss=config["training"].get("loss", "mse"),
            log_interval_batches=config["training"].get("log_interval_batches", 50),
        ),
    )
    history = trainer.fit(dataloader)

    checkpoint_path = Path(config.get("output", {}).get("checkpoint", "artifacts/checkpoints/baseline.pt"))
    _save_checkpoint(checkpoint_path, model, history, model_type="baseline", config=config)
    print(f"Saved checkpoint to {checkpoint_path}")
    return 0


def cmd_train_conditional(config_path: str) -> int:
    config = load_yaml(config_path)
    _, dataloader = _build_dataset_and_loader(config)
    model = ConditionalHRTFEstimator(
        anthropometric_dim=config["model"]["anthropometric_dim"],
        direction_dim=config["model"]["direction_dim"],
        hrtf_dim=config["model"]["hrtf_dim"],
        ear_image_size=config["model"]["ear_image_size"],
        ear_side_dim=config["model"].get("ear_side_dim", 1),
    )
    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            device=config["training"]["device"],
            use_direction=True,
            optimizer=config["training"].get("optimizer", "adam"),
            loss=config["training"].get("loss", "mse"),
            log_interval_batches=config["training"].get("log_interval_batches", 50),
        ),
    )
    history = trainer.fit(dataloader)

    checkpoint_path = Path(config.get("output", {}).get("checkpoint", "artifacts/checkpoints/conditional.pt"))
    _save_checkpoint(checkpoint_path, model, history, model_type="conditional", config=config)
    print(f"Saved checkpoint to {checkpoint_path}")
    return 0


def cmd_evaluate(config_path: str) -> int:
    config = load_yaml(config_path)
    evaluation_config = config["evaluation"]
    dataset = CIPICPreparedDataset(evaluation_config["prepared_root"])
    if len(dataset) == 0:
        raise SystemExit("No prepared samples found.")

    checkpoint = torch.load(evaluation_config["checkpoint"], map_location="cpu")
    model_type = _resolve_model_type(checkpoint, evaluation_config)
    model = _build_model_from_checkpoint(checkpoint, model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=evaluation_config.get("batch_size", 64),
        shuffle=False,
        collate_fn=collate_prepared_samples,
    )

    rmse_values: list[float] = []
    lsd_values: list[float] = []
    per_subject_rmse: dict[str, list[float]] = defaultdict(list)
    per_subject_lsd: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in dataloader:
            if model_type == "conditional":
                prediction = model(batch.anthropometrics, batch.ear_image, batch.direction)
            else:
                prediction = model(batch.anthropometrics, batch.ear_image)
            prediction_np = prediction.numpy()
            reference_np = batch.hrtf.numpy()
            for idx in range(reference_np.shape[0]):
                sample_rmse = rmse(reference_np[idx], prediction_np[idx])
                sample_lsd = log_spectral_distance(reference_np[idx], prediction_np[idx])
                rmse_values.append(sample_rmse)
                lsd_values.append(sample_lsd)
                subject_id = batch.subject_ids[idx]
                per_subject_rmse[subject_id].append(sample_rmse)
                per_subject_lsd[subject_id].append(sample_lsd)

    per_subject = {
        subject_id: {
            "rmse": float(np.mean(subject_rmse)),
            "lsd": float(np.mean(per_subject_lsd[subject_id])),
            "num_samples": len(subject_rmse),
        }
        for subject_id, subject_rmse in sorted(per_subject_rmse.items())
    }
    metrics = {
        "model_type": model_type,
        "num_samples": len(rmse_values),
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values)),
        "lsd_mean": float(np.mean(lsd_values)),
        "lsd_std": float(np.std(lsd_values)),
        "per_subject": per_subject,
    }

    output_path = Path(evaluation_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {output_path}")
    return 0


def cmd_export_sofa(config_path: str) -> int:
    config = load_yaml(config_path)
    hrir = np.zeros((1, 2, 200), dtype=np.float64)
    source_positions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    output = export_simple_free_field_hrir(
        config["export"]["output_sofa"],
        hrir=hrir,
        source_positions_deg=source_positions,
        sampling_rate_hz=config["export"]["sampling_rate"],
    )
    print(f"Wrote SOFA file to {output}")
    return 0


def cmd_measure_anthro(front_image: str, side_image: str, output: str) -> int:
    values = collect_anthropometrics(front_image, side_image, output_path=output)
    print(f"Wrote 17 anthropometric values to {output}")
    print(", ".join(f"{value:.4f}" for value in values))
    return 0


def cmd_predict(
    checkpoint_path: str,
    image_path: str | None,
    left_image_path: str | None,
    right_image_path: str | None,
    image_ear: str,
    anthro: str | None,
    anthro_json: str | None,
    front_image: str | None,
    side_image: str | None,
    measure_output: str | None,
    template_sofa_path: str,
    output_sofa_path: str | None,
    device: str,
) -> int:
    _validate_predict_image_inputs(image_path, left_image_path, right_image_path)
    resolved_device = _resolve_runtime_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model_type = _resolve_model_type(checkpoint, {"model_type": "auto"})
    model = _build_model_from_checkpoint(checkpoint, model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])

    anthropometrics = _resolve_anthropometrics(anthro, anthro_json, front_image, side_image, measure_output)

    saved_config = checkpoint.get("config", {})
    dataset_config = saved_config.get("dataset", {})
    image_size = saved_config.get("model", {}).get("ear_image_size", dataset_config.get("image_size", 64))
    crop = dataset_config.get("crop", {"x": 50, "y": 50, "width": 450, "height": 450})
    edge = dataset_config.get("edge_detection", {"low_threshold": 50, "high_threshold": 100})
    preprocessor = EarImagePreprocessor(
        image_size=image_size,
        crop_x=crop["x"],
        crop_y=crop["y"],
        crop_width=crop["width"],
        crop_height=crop["height"],
        blur_kernel=dataset_config.get("blur_kernel", 5),
        low_threshold=edge["low_threshold"],
        high_threshold=edge["high_threshold"],
    )
    resolved_output_path = _resolve_prediction_output_path(output_sofa_path)
    predictions = predict_sofa_from_image(
        PredictionInputs(
            checkpoint_path=Path(checkpoint_path),
            image_path=Path(image_path) if image_path else None,
            left_image_path=Path(left_image_path) if left_image_path else None,
            right_image_path=Path(right_image_path) if right_image_path else None,
            anthropometrics=anthropometrics,
            template_sofa_path=Path(template_sofa_path),
            output_sofa_path=resolved_output_path,
            image_ear=image_ear,
            model_type=model_type,
            device=resolved_device,
        ),
        model=model,
        image_preprocessor=preprocessor,
    )
    print(f"Wrote predicted SOFA file to {predictions}")
    return 0


def _build_dataset_and_loader(config: dict) -> tuple[CIPICPreparedDataset, DataLoader]:
    dataset = CIPICPreparedDataset(config["dataset"]["prepared_root"])
    if len(dataset) == 0:
        raise SystemExit("No prepared samples found. Run prepare-cipic after dataset preparation.")
    device = config["training"].get("device", "cpu")
    use_cuda = str(device).startswith("cuda")
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_prepared_samples,
        num_workers=config["training"].get("num_workers", 4 if use_cuda else 0),
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )
    return dataset, dataloader


def _save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    history: list[float],
    model_type: str,
    config: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss_history": history,
            "model_type": model_type,
            "config": config,
        },
        checkpoint_path,
    )


def _resolve_model_type(checkpoint: dict, evaluation_config: dict) -> str:
    requested = evaluation_config.get("model_type", "auto")
    if requested != "auto":
        return requested
    return checkpoint.get("model_type", "baseline")


def _resolve_runtime_device(device: str) -> str:
    if device.lower() != "mps":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_prediction_output_path(output_sofa_path: str | None) -> Path:
    if output_sofa_path:
        return Path(output_sofa_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("predictions") / f"prediction-{timestamp}.sofa"


def _validate_predict_image_inputs(
    image_path: str | None,
    left_image_path: str | None,
    right_image_path: str | None,
) -> None:
    has_single_image = image_path is not None
    has_binaural_pair = left_image_path is not None or right_image_path is not None
    if has_single_image and has_binaural_pair:
        raise SystemExit("Provide either --image, or both --left-image and --right-image, but not both modes.")
    if has_single_image:
        return
    if left_image_path and right_image_path:
        return
    raise SystemExit("Provide either --image, or both --left-image and --right-image.")


def _resolve_anthropometrics(
    anthro: str | None,
    anthro_json: str | None,
    front_image: str | None,
    side_image: str | None,
    measure_output: str | None,
) -> np.ndarray:
    if anthro_json:
        values = load_anthropometrics_json(anthro_json)
    elif front_image and side_image:
        values = collect_anthropometrics(front_image, side_image, output_path=measure_output)
    elif anthro:
        values = np.asarray([float(item) for item in anthro.split(",") if item.strip()], dtype=np.float32)
    elif _DEFAULT_ANTHRO_PATH.exists():
        values = load_anthropometrics_json(_DEFAULT_ANTHRO_PATH)
    else:
        raise SystemExit(
            "Provide --anthro, or --anthro-json, or both --front-image and --side-image for interactive measurement. "
            f"No default anthropometric file was found at {_DEFAULT_ANTHRO_PATH}."
        )
    if values.shape[0] != 17:
        raise SystemExit(f"Expected 17 anthropometric values, got {values.shape[0]}.")
    return values


def _build_model_from_checkpoint(checkpoint: dict, model_type: str) -> torch.nn.Module:
    saved_config = checkpoint.get("config", {})
    model_config = saved_config.get("model", {})
    if model_type == "conditional":
        return ConditionalHRTFEstimator(
            anthropometric_dim=model_config.get("anthropometric_dim", 17),
            direction_dim=model_config.get("direction_dim", 2),
            hrtf_dim=model_config.get("hrtf_dim", 200),
            ear_image_size=model_config.get("ear_image_size", 64),
            ear_side_dim=model_config.get("ear_side_dim", 0),
        )
    return BaselineHRTFEstimator(
        anthropometric_dim=model_config.get("anthropometric_dim", 17),
        hrtf_dim=model_config.get("hrtf_dim", 200),
        ear_image_size=model_config.get("ear_image_size", 64),
        ear_side_dim=model_config.get("ear_side_dim", 0),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch-hrtfcnn-assets":
        return cmd_fetch_hrtfcnn_assets(args.config, overwrite=args.overwrite)
    if args.command == "prepare-cipic":
        return cmd_prepare_cipic(args.config)
    if args.command == "train-baseline":
        return cmd_train_baseline(args.config)
    if args.command == "train-conditional":
        return cmd_train_conditional(args.config)
    if args.command == "evaluate":
        return cmd_evaluate(args.config)
    if args.command == "export-sofa":
        return cmd_export_sofa(args.config)
    if args.command == "measure-anthro":
        return cmd_measure_anthro(args.front_image, args.side_image, args.output)
    if args.command == "predict":
        return cmd_predict(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            left_image_path=args.left_image,
            right_image_path=args.right_image,
            image_ear=args.image_ear,
            anthro=args.anthro,
            anthro_json=args.anthro_json,
            front_image=args.front_image,
            side_image=args.side_image,
            measure_output=args.measure_output,
            template_sofa_path=args.template_sofa,
            output_sofa_path=args.output_sofa,
            device=args.device,
        )

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
