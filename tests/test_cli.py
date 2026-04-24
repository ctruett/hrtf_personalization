from __future__ import annotations

from hrtf_personalization.cli.main import build_parser


def test_parser_accepts_train_conditional() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-conditional", "--config", "configs/train-conditional.yaml"])
    assert args.command == "train-conditional"


def test_parser_accepts_fetch_assets() -> None:
    parser = build_parser()
    args = parser.parse_args(["fetch-hrtfcnn-assets", "--config", "configs/dataset.yaml"])
    assert args.command == "fetch-hrtfcnn-assets"


def test_parser_accepts_predict() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--checkpoint",
            "artifacts/checkpoints/conditional.pt",
            "--image",
            "ear.jpg",
            "--image-ear",
            "left",
            "--anthro",
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
            "--template-sofa",
            "/tmp/HRTFCNN/data/template.sofa",
            "--output-sofa",
            "artifacts/predictions/out.sofa",
        ]
    )
    assert args.command == "predict"
    assert args.image_ear == "left"


def test_parser_accepts_measure_anthro() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["measure-anthro", "--front-image", "front.jpg", "--side-image", "side.jpg", "--output", "anthro.json"]
    )
    assert args.command == "measure-anthro"


def test_parser_accepts_predict_with_explicit_left_and_right_images() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--checkpoint",
            "artifacts/checkpoints/conditional.pt",
            "--left-image",
            "left.jpg",
            "--right-image",
            "right.jpg",
            "--anthro",
            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
            "--template-sofa",
            "/tmp/HRTFCNN/data/template.sofa",
        ]
    )
    assert args.command == "predict"
    assert args.left_image == "left.jpg"
    assert args.right_image == "right.jpg"
