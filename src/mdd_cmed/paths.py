import os
from dataclasses import dataclass
from pathlib import Path

from .config import CMEDConfig


@dataclass
class KagglePaths:
    train_root: Path
    public_root: Path
    private_root: Path
    train_meta: Path
    public_meta: Path
    private_meta: Path
    output_root: Path
    processed_dir: Path
    split_dir: Path
    report_dir: Path
    exp_dir: Path
    ckpt_dir: Path
    submission_dir: Path

    @classmethod
    def from_env(cls, cfg: CMEDConfig) -> "KagglePaths":
        train_root = Path(
            os.environ.get(
                "MDD_TRAIN_ROOT",
                "/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-training-set/MDD-Challenge-2025-training-set",
            )
        ).resolve()
        public_root = Path(
            os.environ.get(
                "MDD_PUBLIC_ROOT",
                "/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-public-test/MDD-Challenge-2025-public-test",
            )
        ).resolve()
        private_root = Path(
            os.environ.get(
                "MDD_PRIVATE_ROOT",
                "/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-private-test/MDD-Challenge-2025-private-test",
            )
        ).resolve()
        output_root = Path(os.environ.get("MDD_OUTPUT_ROOT", "/kaggle/working")).resolve()
        processed_dir = output_root / "data" / "processed"
        split_dir = output_root / "data" / "splits"
        report_dir = output_root / "reports"
        exp_dir = output_root / "experiments" / cfg.exp_id
        ckpt_dir = output_root / "checkpoints" / cfg.exp_id
        return cls(
            train_root=train_root,
            public_root=public_root,
            private_root=private_root,
            train_meta=Path(os.environ.get("MDD_TRAIN_META", train_root / "metadata" / "train_phones.csv")),
            public_meta=Path(os.environ.get("MDD_PUBLIC_META", public_root / "metadata" / "public_test_phones.csv")),
            private_meta=Path(os.environ.get("MDD_PRIVATE_META", private_root / "metadata" / "private_test_submission.csv")),
            output_root=output_root,
            processed_dir=processed_dir,
            split_dir=split_dir,
            report_dir=report_dir,
            exp_dir=exp_dir,
            ckpt_dir=ckpt_dir,
            submission_dir=output_root,
        )

    def ensure_output_dirs(self) -> None:
        for path in [
            self.processed_dir,
            self.split_dir,
            self.report_dir,
            self.exp_dir,
            self.ckpt_dir,
            self.submission_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def resolve_audio_path(raw_path: str, roots: list[Path]) -> Path | None:
    raw = Path(str(raw_path))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    for root in roots:
        candidates.append(root / raw)
        candidates.append(root / raw.name)
        if len(raw.parts) >= 2:
            candidates.append(root / raw.parts[-2] / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None
