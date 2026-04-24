from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile


PINNAS_FILE_ID = "1a6LihoO2agENYDM6qdpL9tqUa4yQBHTn"
EAR_FILE_ID = "1rJT5NgX_OoI_5_fFxQmjGvxo6ZgFSvN_"
SOFA_CIPIC_URL = "http://sofacoustics.org/data/database/cipic/"


def fetch_hrtfcnn_assets(repo_root: str | Path, overwrite: bool = False) -> None:
    root = Path(repo_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _fetch_cipic_sofa(root, data_dir, overwrite=overwrite)
    _fetch_google_drive_archives(root, overwrite=overwrite)
    _extract_archives(root, data_dir, overwrite=overwrite)


def _fetch_cipic_sofa(root: Path, data_dir: Path, overwrite: bool) -> None:
    target_dir = data_dir / "cipic_hrtf_sofa"
    if target_dir.exists() and any(target_dir.glob("*.sofa")) and not overwrite:
        return
    target_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "wget",
            "-r",
            "-l1",
            "-w",
            "1",
            "--no-parent",
            SOFA_CIPIC_URL,
        ],
        cwd=root,
    )
    downloaded_dir = root / "sofacoustics.org" / "data" / "database" / "cipic"
    for sofa_file in downloaded_dir.glob("*.sofa"):
        shutil.move(str(sofa_file), target_dir / sofa_file.name)


def _fetch_google_drive_archives(root: Path, overwrite: bool) -> None:
    import gdown

    pinnas_zip = root / "pinnas.zip"
    ear_zip = root / "ear.zip"
    if overwrite or not pinnas_zip.exists():
        gdown.download(id=PINNAS_FILE_ID, output=str(pinnas_zip), quiet=False)
    if overwrite or not ear_zip.exists():
        gdown.download(id=EAR_FILE_ID, output=str(ear_zip), quiet=False)


def _extract_archives(root: Path, data_dir: Path, overwrite: bool) -> None:
    pinnas_dir = data_dir / "CIPIC_hrtf_database"
    ear_dir = data_dir / "ear_photos"
    if overwrite and pinnas_dir.exists():
        shutil.rmtree(pinnas_dir)
    if overwrite and ear_dir.exists():
        shutil.rmtree(ear_dir)

    with ZipFile(root / "pinnas.zip") as archive:
        archive.extractall(root)
    with ZipFile(root / "ear.zip") as archive:
        archive.extractall(root)

    source_pinnas = root / "CIPIC_hrtf_database"
    if source_pinnas.exists() and not pinnas_dir.exists():
        shutil.move(str(source_pinnas), pinnas_dir)

    source_ear = root / "binural-updates" / "ear_photos"
    if source_ear.exists() and not ear_dir.exists():
        shutil.move(str(source_ear), ear_dir)


def _run(cmd: list[str], cwd: Path) -> None:
    process = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        msg = (
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
        raise RuntimeError(msg)
