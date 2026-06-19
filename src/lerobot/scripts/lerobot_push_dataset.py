#!/usr/bin/env python

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


@dataclass
class PushDatasetConfig:
    repo_id: str
    root: str | Path

    private: bool = False
    branch: str | None = None
    tags: list[str] | None = None
    license: str | None = "apache-2.0"
    tag_version: bool = True
    push_videos: bool = True
    upload_large_folder: bool = False

    # Useful behind corporate networks with TLS interception.
    ca_bundle: str | None = None


def check_lerobot_dataset_root(root: Path) -> None:
    required_paths = [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        root / "data",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Invalid LeRobot dataset root. Missing:\n" + "\n".join(missing)
        )


@parser.wrap()
def push_dataset(cfg: PushDatasetConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = cfg.ca_bundle
        os.environ["CURL_CA_BUNDLE"] = cfg.ca_bundle
        os.environ["SSL_CERT_FILE"] = cfg.ca_bundle

    root = Path(cfg.root).expanduser().resolve()
    check_lerobot_dataset_root(root)

    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=root,
        download_videos=False,
    )

    logging.info(dataset)
    logging.info(f"Pushing local dataset {root} to Hugging Face dataset repo {cfg.repo_id}")

    dataset.push_to_hub(
        branch=cfg.branch,
        tags=cfg.tags,
        license=cfg.license,
        tag_version=cfg.tag_version,
        push_videos=cfg.push_videos,
        private=cfg.private,
        upload_large_folder=cfg.upload_large_folder,
    )

    logging.info("Dataset successfully pushed to the Hugging Face Hub.")


def main() -> None:
    push_dataset()


if __name__ == "__main__":
    main()