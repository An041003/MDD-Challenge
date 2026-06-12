import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "MDD_CMED_Kaggle.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # C-MED V1: Canonical-conditioned Mispronunciation Edit Decoder

        This notebook is a clean replacement for the old CTC-first pipeline.
        It does not generate a full ASR transcript and then patch it back to
        canonical. Instead, it conditions on the canonical phoneme sequence and
        predicts token-level edits directly.

        Allowed pretrained speech backbone:

        `nguyenvulebinh/wav2vec2-base-vietnamese-250h`

        V1 implements `KEEP` and `SUBSTITUTE` only. DELETE and INSERT are
        logged but not trained until V1 passes the tiny overfit gate and shows
        non-collapsed validation behavior.
        """
    ),
    md(
        """
        ## 0. Kaggle runtime contract

        This notebook is Kaggle-only. It expects these mounted input roots:

        - `/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-training-set/MDD-Challenge-2025-training-set`
        - `/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-public-test/MDD-Challenge-2025-public-test`
        - `/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-private-test/MDD-Challenge-2025-private-test`

        All generated artifacts are written under `/kaggle/working`.

        Full training is disabled until the tiny balanced overfit gate passes.
        """
    ),
    code(
        r"""
        # Optional dependency install for fresh Kaggle runtimes.
        # Uncomment only when the runtime is missing packages.
        # !pip install -q pandas numpy scikit-learn tqdm pyyaml torch torchaudio transformers

        import json
        import hashlib
        import math
        import os
        import random
        import re
        import shutil
        import zipfile
        from dataclasses import asdict, dataclass
        from pathlib import Path
        from typing import Dict, Iterable, List, Optional, Sequence, Tuple

        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchaudio
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from tqdm.auto import tqdm
        from transformers import AutoModel, AutoProcessor

        BACKBONE_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
        assert BACKBONE_NAME == "nguyenvulebinh/wav2vec2-base-vietnamese-250h"

        SEED = 2026
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", DEVICE)
        """
    ),
    code(
        r"""
        @dataclass
        class CFG:
            exp_id: str = "cmed_v1_keep_sub"
            sample_rate: int = 16000
            model_dim: int = 256
            canon_layers: int = 2
            num_heads: int = 4
            dropout: float = 0.15
            canonical_dropout: float = 0.12
            max_canonical_len: int = 256
            train_batch_size: int = 4
            eval_batch_size: int = 8
            num_workers: int = 0
            tiny_epochs: int = 40
            stage_a_epochs: int = 20
            replacement_head_epochs: int = 5
            head_lr: float = 2e-4
            replacement_head_lr: float = 5e-4
            weight_decay: float = 0.01
            grad_clip: float = 1.0
            detection_focal_gamma: float = 2.0
            detection_loss_weight: float = 1.5
            operation_loss_weight: float = 1.0
            replacement_loss_weight: float = 1.0
            utterance_loss_weight: float = 0.5
            min_recall: float = 0.20
            min_true_reject: int = 10
            max_canonical_copy_rate: float = 0.95
            max_per: float = 0.10
            max_der: float = 0.35
            min_correct_diagnosis: int = 10
            min_fold_correct_diagnosis: int = 1
            calibration_folds: int = 5
            default_sub_threshold: float = 0.50
            default_replacement_threshold: float = 0.50

        cfg = CFG()

        KAGGLE_TRAIN_ROOT = Path(os.environ.get(
            "MDD_TRAIN_ROOT",
            "/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-training-set/MDD-Challenge-2025-training-set",
        )).resolve()
        PUBLIC_ROOT = Path(os.environ.get(
            "MDD_PUBLIC_ROOT",
            "/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-public-test/MDD-Challenge-2025-public-test",
        )).resolve()
        PRIVATE_ROOT = Path(os.environ.get(
            "MDD_PRIVATE_ROOT",
            "/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-private-test/MDD-Challenge-2025-private-test",
        )).resolve()

        DATA_ROOT = KAGGLE_TRAIN_ROOT
        PROJECT_ROOT = Path("/kaggle/working").resolve()

        TRAIN_META_PATH = Path(os.environ.get("MDD_TRAIN_META", DATA_ROOT / "metadata" / "train_phones.csv"))
        PUBLIC_META_PATH = Path(os.environ.get("MDD_PUBLIC_META", PUBLIC_ROOT / "metadata" / "public_test_phones.csv"))
        PRIVATE_META_PATH = Path(os.environ.get("MDD_PRIVATE_META", PRIVATE_ROOT / "metadata" / "private_test_submission.csv"))
        PRIVATE_EXAMPLE_PATH = PRIVATE_ROOT / "metadata" / "private_test_submission_example.csv"

        OUTPUT_ROOT = Path(os.environ.get("MDD_OUTPUT_ROOT", "/kaggle/working")).resolve()
        PROCESSED_DIR = OUTPUT_ROOT / "data" / "processed"
        SPLIT_DIR = OUTPUT_ROOT / "data" / "splits"
        REPORT_DIR = OUTPUT_ROOT / "reports"
        EXP_DIR = OUTPUT_ROOT / "experiments" / cfg.exp_id
        CKPT_DIR = OUTPUT_ROOT / "checkpoints" / cfg.exp_id
        SUBMISSION_DIR = OUTPUT_ROOT

        for d in [PROCESSED_DIR, SPLIT_DIR, REPORT_DIR, EXP_DIR, CKPT_DIR, SUBMISSION_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("DATA_ROOT:", DATA_ROOT)
        print("TRAIN_META_PATH:", TRAIN_META_PATH)
        print("PUBLIC_META_PATH:", PUBLIC_META_PATH)
        print("PRIVATE_META_PATH:", PRIVATE_META_PATH)
        print("OUTPUT_ROOT:", OUTPUT_ROOT)
        """
    ),
    md(
        """
        ## 1. Data audit

        Stop if the training metadata is missing, required columns are absent,
        or audio paths cannot be resolved.
        """
    ),
    code(
        r"""
        REQUIRED_TRAIN_COLUMNS = {"id", "path", "canonical", "transcript"}
        REQUIRED_PRIVATE_COLUMNS = {"id", "path", "canonical"}

        def read_csv_checked(path: Path, required_cols: set, name: str) -> pd.DataFrame:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
            df = pd.read_csv(path)
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{name} missing columns: {sorted(missing)}")
            return df

        def split_phones(text) -> List[str]:
            if pd.isna(text):
                return []
            return str(text).strip().split()

        def resolve_audio_path(raw_path: str, roots: Sequence[Path]) -> Optional[Path]:
            raw = Path(str(raw_path))
            candidates = []
            if raw.is_absolute():
                candidates.append(raw)
            for root in roots:
                candidates.append(root / raw)
                candidates.append(root / raw.name)
                if len(raw.parts) >= 2:
                    candidates.append(root / raw.parts[-2] / raw.name)
                candidates.append(root / "audio_data" / raw.name)
                candidates.append(root / "audio_data" / "train" / raw.name)
                candidates.append(root / "audio_data" / "public_test" / raw.name)
                candidates.append(root / "audio_data" / "private_test" / raw.name)
            for c in candidates:
                if c.exists():
                    return c.resolve()
            return None

        train_df = read_csv_checked(TRAIN_META_PATH, REQUIRED_TRAIN_COLUMNS, "train_phones.csv")
        train_df["canonical_tokens"] = train_df["canonical"].map(split_phones)
        train_df["transcript_tokens"] = train_df["transcript"].map(split_phones)
        train_df["is_error"] = train_df["canonical"].astype(str) != train_df["transcript"].astype(str)

        train_audio_roots = [DATA_ROOT, TRAIN_META_PATH.parent.parent, PROJECT_ROOT]
        train_df["audio_path_resolved"] = train_df["path"].map(lambda p: resolve_audio_path(p, train_audio_roots))

        missing_audio = train_df[train_df["audio_path_resolved"].isna()]
        print("train rows:", len(train_df))
        print("correct utterances:", int((~train_df["is_error"]).sum()))
        print("mispronounced utterances:", int(train_df["is_error"].sum()))
        print("missing audio:", len(missing_audio))
        if len(missing_audio):
            display(missing_audio[["id", "path"]].head(20))
            raise FileNotFoundError("Some train audio paths cannot be resolved.")

        audit = {
            "rows": int(len(train_df)),
            "correct_utterances": int((~train_df["is_error"]).sum()),
            "mispronounced_utterances": int(train_df["is_error"].sum()),
            "unique_canonical_tokens": int(len(set(t for xs in train_df["canonical_tokens"] for t in xs))),
            "unique_transcript_tokens": int(len(set(t for xs in train_df["transcript_tokens"] for t in xs))),
        }
        (REPORT_DIR / "cmed_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        audit
        """
    ),
    md(
        """
        ## 2. Speaker-safe split

        Splits are grouped by speaker prefix. Calibration is kept separate from
        validation so thresholds are never tuned on final validation.
        """
    ),
    code(
        r"""
        def parse_speaker_prefix(sample_id: str) -> str:
            s = str(sample_id)
            m = re.match(r"^(.*?)[_-]S\d+", s)
            if m:
                return m.group(1)
            parts = s.split("_")
            if len(parts) >= 3:
                return "_".join(parts[:3])
            parts = s.split("-")
            if len(parts) >= 2:
                return "-".join(parts[:-1])
            return s

        train_df["speaker_prefix"] = train_df["id"].map(parse_speaker_prefix)

        def make_speaker_safe_splits(df: pd.DataFrame, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            groups = df["speaker_prefix"].astype(str)
            n_groups = groups.nunique()
            if n_groups < 3:
                raise ValueError(f"Cannot create speaker-safe train/calibration/validation split with only {n_groups} speaker groups.")

            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
            train_idx, temp_idx = next(gss1.split(df, groups=groups))
            train_part = df.iloc[train_idx].copy()
            temp_part = df.iloc[temp_idx].copy()

            temp_groups = temp_part["speaker_prefix"].astype(str)
            if temp_groups.nunique() < 2:
                raise ValueError("Temporary split has too few speaker groups for calibration/validation.")

            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
            cal_rel_idx, val_rel_idx = next(gss2.split(temp_part, groups=temp_groups))
            cal_part = temp_part.iloc[cal_rel_idx].copy()
            val_part = temp_part.iloc[val_rel_idx].copy()

            train_sp = set(train_part["speaker_prefix"])
            cal_sp = set(cal_part["speaker_prefix"])
            val_sp = set(val_part["speaker_prefix"])
            assert train_sp.isdisjoint(cal_sp)
            assert train_sp.isdisjoint(val_sp)
            assert cal_sp.isdisjoint(val_sp)
            return train_part, cal_part, val_part

        split_train_df, split_cal_df, split_val_df = make_speaker_safe_splits(train_df)

        for name, part in [("train", split_train_df), ("calibration", split_cal_df), ("validation", split_val_df)]:
            out_path = SPLIT_DIR / f"{name}.csv"
            part.drop(columns=["canonical_tokens", "transcript_tokens"], errors="ignore").to_csv(out_path, index=False)
            print(name, "rows:", len(part), "speakers:", part["speaker_prefix"].nunique(), "error_rate:", round(float(part["is_error"].mean()), 4))
        """
    ),
    md(
        """
        ## 3. Phoneme vocab

        The vocab is built from training canonical/transcript tokens and any
        available test canonical tokens. Unknowns are still handled explicitly.
        """
    ),
    code(
        r"""
        PAD_TOKEN = "<pad>"
        UNK_TOKEN = "<unk>"
        SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

        def iter_phone_tokens_from_df(df: pd.DataFrame, columns: Sequence[str]) -> Iterable[str]:
            for col in columns:
                if col not in df.columns:
                    continue
                for text in df[col].fillna(""):
                    yield from split_phones(text)

        vocab_tokens = set(iter_phone_tokens_from_df(train_df, ["canonical", "transcript"]))

        if PUBLIC_META_PATH.exists():
            public_vocab_df = pd.read_csv(PUBLIC_META_PATH)
            vocab_tokens.update(iter_phone_tokens_from_df(public_vocab_df, ["canonical", "transcript"]))
        if PRIVATE_META_PATH.exists():
            private_vocab_df = pd.read_csv(PRIVATE_META_PATH)
            vocab_tokens.update(iter_phone_tokens_from_df(private_vocab_df, ["canonical"]))

        id_to_token = SPECIAL_TOKENS + sorted(vocab_tokens)
        token_to_id = {tok: idx for idx, tok in enumerate(id_to_token)}
        pad_id = token_to_id[PAD_TOKEN]
        unk_id = token_to_id[UNK_TOKEN]

        vocab = {"id_to_token": id_to_token, "token_to_id": token_to_id, "pad_id": pad_id, "unk_id": unk_id}
        (PROCESSED_DIR / "phoneme_vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
        print("vocab size:", len(id_to_token))
        print(id_to_token[:20])
        """
    ),
    md(
        """
        ## 4. Levenshtein alignment and V1 labels

        V1 learns `KEEP` and `SUBSTITUTE`. Deletions and insertions are
        tracked in reports, but ignored by V1 labels so they cannot silently
        turn into sentence-level canonical fallback.
        """
    ),
    code(
        r"""
        OP_KEEP = 0
        OP_SUB = 1
        OP_NAMES = {OP_KEEP: "KEEP", OP_SUB: "SUBSTITUTE"}
        IGNORE_INDEX = -100

        def levenshtein_alignment(src: Sequence[str], tgt: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
            n, m = len(src), len(tgt)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            back = [[None] * (m + 1) for _ in range(n + 1)]

            for i in range(1, n + 1):
                dp[i][0] = i
                back[i][0] = "delete"
            for j in range(1, m + 1):
                dp[0][j] = j
                back[0][j] = "insert"

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if src[i - 1] == tgt[j - 1]:
                        best = (dp[i - 1][j - 1], "equal")
                    else:
                        best = (dp[i - 1][j - 1] + 1, "substitute")
                    cand_delete = (dp[i - 1][j] + 1, "delete")
                    cand_insert = (dp[i][j - 1] + 1, "insert")
                    best = min([best, cand_delete, cand_insert], key=lambda x: (x[0], {"equal": 0, "substitute": 1, "delete": 2, "insert": 3}[x[1]]))
                    dp[i][j], back[i][j] = best

            align = []
            i, j = n, m
            while i > 0 or j > 0:
                op = back[i][j]
                if op == "equal":
                    align.append(("equal", src[i - 1], tgt[j - 1]))
                    i -= 1
                    j -= 1
                elif op == "substitute":
                    align.append(("substitute", src[i - 1], tgt[j - 1]))
                    i -= 1
                    j -= 1
                elif op == "delete":
                    align.append(("delete", src[i - 1], None))
                    i -= 1
                elif op == "insert":
                    align.append(("insert", None, tgt[j - 1]))
                    j -= 1
                else:
                    raise RuntimeError("Invalid alignment backtrace.")
            align.reverse()
            return align

        def make_v1_label_row(row: pd.Series) -> dict:
            canonical_tokens = split_phones(row["canonical"])
            transcript_tokens = split_phones(row["transcript"])
            align = levenshtein_alignment(canonical_tokens, transcript_tokens)

            canonical_ids = []
            operation_labels = []
            detection_labels = []
            replacement_targets = []
            ignored_insertions = []
            ignored_deletions = []

            canon_pos = 0
            for op, src_tok, tgt_tok in align:
                if op == "insert":
                    ignored_insertions.append({"gap_after": canon_pos - 1, "token": tgt_tok})
                    continue

                canonical_ids.append(token_to_id.get(src_tok, unk_id))
                if op == "equal":
                    operation_labels.append(OP_KEEP)
                    detection_labels.append(0)
                    replacement_targets.append(IGNORE_INDEX)
                elif op == "substitute":
                    operation_labels.append(OP_SUB)
                    detection_labels.append(1)
                    replacement_targets.append(token_to_id.get(tgt_tok, unk_id))
                elif op == "delete":
                    operation_labels.append(IGNORE_INDEX)
                    detection_labels.append(IGNORE_INDEX)
                    replacement_targets.append(IGNORE_INDEX)
                    ignored_deletions.append({"position": canon_pos, "token": src_tok})
                else:
                    raise ValueError(op)
                canon_pos += 1

            assert len(canonical_ids) == len(canonical_tokens)
            utterance_label = int(any(x == 1 for x in detection_labels))
            return {
                "canonical_ids": canonical_ids,
                "operation_labels": operation_labels,
                "detection_labels": detection_labels,
                "replacement_targets": replacement_targets,
                "utterance_label": utterance_label,
                "ignored_insertions": ignored_insertions,
                "ignored_deletions": ignored_deletions,
            }

        label_records = []
        insertion_records = []
        deletion_records = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="building V1 labels"):
            lab = make_v1_label_row(row)
            rec = {
                "id": row["id"],
                "canonical": row["canonical"],
                "transcript": row["transcript"],
                "canonical_ids": json.dumps(lab["canonical_ids"]),
                "operation_labels": json.dumps(lab["operation_labels"]),
                "detection_labels": json.dumps(lab["detection_labels"]),
                "replacement_targets": json.dumps(lab["replacement_targets"]),
                "utterance_label": lab["utterance_label"],
            }
            label_records.append(rec)
            for x in lab["ignored_insertions"]:
                insertion_records.append({"id": row["id"], **x})
            for x in lab["ignored_deletions"]:
                deletion_records.append({"id": row["id"], **x})

        labels_df = pd.DataFrame(label_records)
        labels_df.to_csv(PROCESSED_DIR / "cmed_v1_edit_labels.csv", index=False)
        pd.DataFrame(insertion_records).to_csv(REPORT_DIR / "ignored_insertions.csv", index=False)
        pd.DataFrame(deletion_records).to_csv(REPORT_DIR / "ignored_deletions_v1.csv", index=False)

        def count_label_values(json_col: pd.Series) -> Dict[int, int]:
            counts = {}
            for text in json_col:
                for v in json.loads(text):
                    counts[int(v)] = counts.get(int(v), 0) + 1
            return counts

        label_summary = {
            "operation_counts": count_label_values(labels_df["operation_labels"]),
            "detection_counts": count_label_values(labels_df["detection_labels"]),
            "ignored_insertions": len(insertion_records),
            "ignored_deletions": len(deletion_records),
        }
        (REPORT_DIR / "edit_label_distribution.json").write_text(json.dumps(label_summary, indent=2), encoding="utf-8")
        label_summary
        """
    ),
    md(
        """
        ## 5. Dataset and collate

        Audio is loaded with torchaudio, resampled to 16 kHz, and encoded by
        the Vietnamese wav2vec2 feature extractor. Canonical tokens are padded
        separately from audio frames.
        """
    ),
    code(
        r"""
        labels_lookup = labels_df.set_index("id").to_dict("index")

        class CMEDDataset(Dataset):
            def __init__(self, frame: pd.DataFrame, labels_lookup: Optional[Dict[str, dict]] = None, has_labels: bool = True):
                self.df = frame.reset_index(drop=True).copy()
                self.labels_lookup = labels_lookup
                self.has_labels = has_labels

            def __len__(self):
                return len(self.df)

            def _load_audio(self, path: Path) -> torch.Tensor:
                wav, sr = torchaudio.load(str(path))
                wav = wav.mean(dim=0)
                if sr != cfg.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
                return wav

            def __getitem__(self, idx: int) -> dict:
                row = self.df.iloc[idx]
                audio_path = row.get("audio_path_resolved")
                if audio_path is None or pd.isna(audio_path):
                    raise FileNotFoundError(f"Missing resolved audio path for {row['id']}")
                wav = self._load_audio(Path(audio_path))
                canonical_tokens = split_phones(row["canonical"])
                canonical_ids = [token_to_id.get(t, unk_id) for t in canonical_tokens]

                item = {
                    "id": row["id"],
                    "path": row["path"],
                    "canonical": row["canonical"],
                    "canonical_tokens": canonical_tokens,
                    "canonical_ids": torch.tensor(canonical_ids, dtype=torch.long),
                    "waveform": wav,
                }
                if "transcript" in row:
                    item["transcript"] = row["transcript"]

                if self.has_labels:
                    lab = self.labels_lookup[row["id"]]
                    item.update({
                        "operation_labels": torch.tensor(json.loads(lab["operation_labels"]), dtype=torch.long),
                        "detection_labels": torch.tensor(json.loads(lab["detection_labels"]), dtype=torch.long),
                        "replacement_targets": torch.tensor(json.loads(lab["replacement_targets"]), dtype=torch.long),
                        "utterance_label": torch.tensor(float(lab["utterance_label"]), dtype=torch.float),
                    })
                return item

        processor = AutoProcessor.from_pretrained(BACKBONE_NAME)

        def pad_1d_tensors(values: List[torch.Tensor], pad_value: int, dtype=torch.long) -> Tuple[torch.Tensor, torch.Tensor]:
            max_len = max(v.numel() for v in values)
            out = torch.full((len(values), max_len), pad_value, dtype=dtype)
            mask = torch.zeros((len(values), max_len), dtype=torch.bool)
            for i, v in enumerate(values):
                out[i, : v.numel()] = v.to(dtype=dtype)
                mask[i, : v.numel()] = True
            return out, mask

        def cmed_collate(batch: List[dict]) -> dict:
            waves = [b["waveform"].numpy() for b in batch]
            audio = processor(waves, sampling_rate=cfg.sample_rate, return_tensors="pt", padding=True)
            canonical_ids, canonical_mask = pad_1d_tensors([b["canonical_ids"] for b in batch], pad_id, torch.long)

            out = {
                "ids": [b["id"] for b in batch],
                "paths": [b["path"] for b in batch],
                "canonical": [b["canonical"] for b in batch],
                "canonical_tokens": [b["canonical_tokens"] for b in batch],
                "input_values": audio["input_values"],
                "audio_attention_mask": audio.get("attention_mask"),
                "canonical_ids": canonical_ids,
                "canonical_mask": canonical_mask,
            }
            if "transcript" in batch[0]:
                out["transcript"] = [b.get("transcript") for b in batch]
            if "operation_labels" in batch[0]:
                out["operation_labels"], _ = pad_1d_tensors([b["operation_labels"] for b in batch], IGNORE_INDEX, torch.long)
                out["detection_labels"], _ = pad_1d_tensors([b["detection_labels"] for b in batch], IGNORE_INDEX, torch.long)
                out["replacement_targets"], _ = pad_1d_tensors([b["replacement_targets"] for b in batch], IGNORE_INDEX, torch.long)
                out["utterance_labels"] = torch.stack([b["utterance_label"] for b in batch])
            return out

        def move_batch_to_device(batch: dict, device: torch.device) -> dict:
            out = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device)
                else:
                    out[k] = v
            return out
        """
    ),
    md(
        """
        ## 6. C-MED V1 model

        Canonical phoneme states query Vietnamese wav2vec2 audio states through
        cross-attention. Heads predict token detection, operation, replacement,
        and utterance-level error.
        """
    ),
    code(
        r"""
        class CMEDV1Model(nn.Module):
            def __init__(self, vocab_size: int, pad_id: int, unk_id: int):
                super().__init__()
                self.pad_id = pad_id
                self.unk_id = unk_id
                self.audio_encoder = AutoModel.from_pretrained(BACKBONE_NAME)
                audio_dim = self.audio_encoder.config.hidden_size

                self.audio_proj = nn.Linear(audio_dim, cfg.model_dim)
                self.phone_embedding = nn.Embedding(vocab_size, cfg.model_dim, padding_idx=pad_id)
                self.position_embedding = nn.Embedding(cfg.max_canonical_len, cfg.model_dim)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.model_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.model_dim * 4,
                    dropout=cfg.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.canonical_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.canon_layers)
                self.cross_attention = nn.MultiheadAttention(cfg.model_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
                self.fuse_norm = nn.LayerNorm(cfg.model_dim)
                self.fuse_ffn = nn.Sequential(
                    nn.Linear(cfg.model_dim, cfg.model_dim * 4),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.model_dim * 4, cfg.model_dim),
                    nn.Dropout(cfg.dropout),
                )
                self.out_norm = nn.LayerNorm(cfg.model_dim)

                self.detection_head = nn.Linear(cfg.model_dim, 2)
                self.operation_head = nn.Linear(cfg.model_dim, 2)
                self.replacement_head = nn.Linear(cfg.model_dim, vocab_size)
                self.utterance_head = nn.Sequential(
                    nn.Linear(cfg.model_dim, cfg.model_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.model_dim, 1),
                )

            def freeze_audio_encoder(self):
                for p in self.audio_encoder.parameters():
                    p.requires_grad = False

            def freeze_except_replacement_head(self):
                for p in self.parameters():
                    p.requires_grad = False
                for p in self.replacement_head.parameters():
                    p.requires_grad = True

            def _audio_feature_mask(self, feature_len: int, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if attention_mask is None:
                    return None
                if hasattr(self.audio_encoder, "_get_feature_vector_attention_mask"):
                    return self.audio_encoder._get_feature_vector_attention_mask(feature_len, attention_mask)
                return None

            def _apply_canonical_dropout(self, canonical_ids: torch.Tensor, canonical_mask: torch.Tensor) -> torch.Tensor:
                if not self.training or cfg.canonical_dropout <= 0:
                    return canonical_ids
                drop = torch.rand_like(canonical_ids.float()) < cfg.canonical_dropout
                drop = drop & canonical_mask & (canonical_ids != self.pad_id)
                out = canonical_ids.clone()
                out[drop] = self.unk_id
                return out

            def forward(
                self,
                input_values: torch.Tensor,
                audio_attention_mask: Optional[torch.Tensor],
                canonical_ids: torch.Tensor,
                canonical_mask: torch.Tensor,
            ) -> dict:
                audio_out = self.audio_encoder(input_values=input_values, attention_mask=audio_attention_mask)
                audio_states = self.audio_proj(audio_out.last_hidden_state)
                audio_feature_mask = self._audio_feature_mask(audio_states.shape[1], audio_attention_mask)
                audio_key_padding_mask = None if audio_feature_mask is None else ~audio_feature_mask.bool()

                canonical_ids = self._apply_canonical_dropout(canonical_ids, canonical_mask)
                bsz, seq_len = canonical_ids.shape
                if seq_len > cfg.max_canonical_len:
                    raise ValueError(f"canonical length {seq_len} exceeds max_canonical_len={cfg.max_canonical_len}")
                positions = torch.arange(seq_len, device=canonical_ids.device).unsqueeze(0).expand(bsz, seq_len)
                canon_states = self.phone_embedding(canonical_ids) + self.position_embedding(positions)
                canon_states = self.canonical_encoder(canon_states, src_key_padding_mask=~canonical_mask.bool())

                attended_audio, attn_weights = self.cross_attention(
                    query=canon_states,
                    key=audio_states,
                    value=audio_states,
                    key_padding_mask=audio_key_padding_mask,
                    need_weights=False,
                )
                fused = self.fuse_norm(canon_states + attended_audio)
                fused = self.out_norm(fused + self.fuse_ffn(fused))

                masked_fused = fused * canonical_mask.unsqueeze(-1).float()
                pooled = masked_fused.sum(dim=1) / canonical_mask.sum(dim=1, keepdim=True).clamp(min=1).float()

                return {
                    "detection_logits": self.detection_head(fused),
                    "operation_logits": self.operation_head(fused),
                    "replacement_logits": self.replacement_head(fused),
                    "utterance_logits": self.utterance_head(pooled).squeeze(-1),
                }
        """
    ),
    md(
        """
        ## 7. Losses and anti-collapse weights

        Detection uses focal loss with class balance. Operation and replacement
        are token-level losses with ignore masks for unsupported V1 edit types.
        Replacement CE is computed only on gold substitution positions. After
        Stage A, the notebook can run a replacement-head-only refinement phase
        without changing the model architecture or checkpoint state dict.
        """
    ),
    code(
        r"""
        def flatten_valid(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX):
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            valid = flat_targets != ignore_index
            return flat_logits[valid], flat_targets[valid]

        def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> torch.Tensor:
            logits, targets = flatten_valid(logits, targets)
            if targets.numel() == 0:
                return logits.sum() * 0.0
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            idx = torch.arange(targets.numel(), device=targets.device)
            pt = probs[idx, targets]
            log_pt = log_probs[idx, targets]
            loss = -((1 - pt) ** gamma) * log_pt
            if alpha is not None:
                loss = loss * alpha.to(targets.device)[targets]
            return loss.mean()

        def ce_token_loss(logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
            logits, targets = flatten_valid(logits, targets)
            if targets.numel() == 0:
                return logits.sum() * 0.0
            return F.cross_entropy(logits, targets, weight=None if weight is None else weight.to(logits.device))

        def class_weights_from_json_labels(frame: pd.DataFrame, labels_lookup: Dict[str, dict], key: str, num_classes: int) -> torch.Tensor:
            counts = torch.ones(num_classes, dtype=torch.float)
            for sample_id in frame["id"]:
                vals = json.loads(labels_lookup[sample_id][key])
                for v in vals:
                    if v != IGNORE_INDEX:
                        counts[int(v)] += 1
            inv = counts.sum() / counts
            weights = inv / inv.mean()
            return weights

        detection_alpha = class_weights_from_json_labels(split_train_df, labels_lookup, "detection_labels", 2)
        operation_weights = class_weights_from_json_labels(split_train_df, labels_lookup, "operation_labels", 2)
        utt_pos = max(float(split_train_df["is_error"].sum()), 1.0)
        utt_neg = max(float((~split_train_df["is_error"]).sum()), 1.0)
        utterance_pos_weight = torch.tensor([utt_neg / utt_pos], dtype=torch.float)
        print("detection_alpha:", detection_alpha.tolist())
        print("operation_weights:", operation_weights.tolist())
        print("utterance_pos_weight:", utterance_pos_weight.item())

        def compute_cmed_loss(outputs: dict, batch: dict) -> Tuple[torch.Tensor, dict]:
            detection_loss = focal_loss(
                outputs["detection_logits"],
                batch["detection_labels"],
                alpha=detection_alpha,
                gamma=cfg.detection_focal_gamma,
            )
            operation_loss = ce_token_loss(outputs["operation_logits"], batch["operation_labels"], weight=operation_weights)
            replacement_loss = ce_token_loss(outputs["replacement_logits"], batch["replacement_targets"], weight=None)
            utterance_loss = F.binary_cross_entropy_with_logits(
                outputs["utterance_logits"],
                batch["utterance_labels"],
                pos_weight=utterance_pos_weight.to(outputs["utterance_logits"].device),
            )
            total = (
                cfg.detection_loss_weight * detection_loss
                + cfg.operation_loss_weight * operation_loss
                + cfg.replacement_loss_weight * replacement_loss
                + cfg.utterance_loss_weight * utterance_loss
            )
            logs = {
                "loss": float(total.detach().cpu()),
                "detection_loss": float(detection_loss.detach().cpu()),
                "operation_loss": float(operation_loss.detach().cpu()),
                "replacement_loss": float(replacement_loss.detach().cpu()),
                "utterance_loss": float(utterance_loss.detach().cpu()),
            }
            return total, logs
        """
    ),
    md(
        """
        ## 8. Decoding and official-style metrics

        Decoding filters each edit independently. There is no whole-sentence
        canonical fallback.
        """
    ),
    code(
        r"""
        def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
            n, m = len(a), len(b)
            dp = list(range(m + 1))
            for i in range(1, n + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, m + 1):
                    old = dp[j]
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                    prev = old
            return dp[m]

        def token_errors_against_canonical(canonical: Sequence[str], other: Sequence[str]) -> Dict[int, str]:
            errors = {}
            align = levenshtein_alignment(canonical, other)
            pos = -1
            for op, src, tgt in align:
                if op == "insert":
                    errors[pos] = tgt
                    continue
                pos += 1
                if op == "substitute":
                    errors[pos] = tgt
                elif op == "delete":
                    errors[pos] = "<deleted>"
            return errors

        def decode_batch(outputs: dict, batch: dict, sub_threshold: float, replacement_threshold: float) -> List[str]:
            op_probs = F.softmax(outputs["operation_logits"], dim=-1).detach().cpu()
            repl_probs = F.softmax(outputs["replacement_logits"], dim=-1).detach().cpu()
            repl_conf, repl_ids = repl_probs.max(dim=-1)
            canonical_mask = batch["canonical_mask"].detach().cpu()
            predicts = []
            for b, tokens in enumerate(batch["canonical_tokens"]):
                pred_tokens = []
                for i, tok in enumerate(tokens):
                    if not bool(canonical_mask[b, i]):
                        continue
                    op = int(op_probs[b, i].argmax())
                    conf = float(op_probs[b, i, OP_SUB])
                    repl_token_conf = float(repl_conf[b, i])
                    repl = id_to_token[int(repl_ids[b, i])]
                    if (
                        op == OP_SUB
                        and conf >= sub_threshold
                        and repl_token_conf >= replacement_threshold
                        and repl not in SPECIAL_TOKENS
                        and repl != tok
                    ):
                        pred_tokens.append(repl)
                    else:
                        pred_tokens.append(tok)
                predicts.append(" ".join(pred_tokens))
            return predicts

        def mdd_metrics_from_sequences(canonicals: Sequence[str], transcripts: Sequence[str], predicts: Sequence[str]) -> dict:
            tp = fp = fn = 0
            correct_diag = wrong_diag = 0
            per_values = []
            copy_count = 0
            pred_edit_counts = []

            for canonical, transcript, predict in zip(canonicals, transcripts, predicts):
                c = split_phones(canonical)
                t = split_phones(transcript)
                p = split_phones(predict)
                actual_errors = token_errors_against_canonical(c, t)
                pred_errors = token_errors_against_canonical(c, p)

                actual_pos = set(actual_errors)
                pred_pos = set(pred_errors)
                tp += len(actual_pos & pred_pos)
                fp += len(pred_pos - actual_pos)
                fn += len(actual_pos - pred_pos)
                for pos in actual_pos & pred_pos:
                    if pred_errors[pos] == actual_errors[pos]:
                        correct_diag += 1
                    else:
                        wrong_diag += 1

                per_values.append(edit_distance(p, t) / max(1, len(t)))
                copy_count += int(p == c)
                pred_edit_counts.append(len(pred_pos))

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            der = wrong_diag / max(1, correct_diag + wrong_diag)
            per = float(np.mean(per_values)) if per_values else 0.0
            score = 0.5 * f1 + 0.4 * (1 - der) + 0.1 * (1 - per)
            return {
                "F1": f1,
                "DER": der,
                "PER": per,
                "Score": score,
                "precision": precision,
                "recall": recall,
                "true_reject": tp,
                "false_reject": fp,
                "false_accept": fn,
                "correct_diagnosis": correct_diag,
                "wrong_diagnosis": wrong_diag,
                "canonical_copy_rate": copy_count / max(1, len(canonicals)),
                "avg_predicted_edits_per_utterance": float(np.mean(pred_edit_counts)) if pred_edit_counts else 0.0,
                "zero_edit_prediction_rate": float(np.mean([x == 0 for x in pred_edit_counts])) if pred_edit_counts else 0.0,
            }

        def token_supervised_metrics(outputs: dict, batch: dict) -> dict:
            det_pred = outputs["detection_logits"].argmax(dim=-1)
            op_pred = outputs["operation_logits"].argmax(dim=-1)
            repl_pred = outputs["replacement_logits"].argmax(dim=-1)

            det_target = batch["detection_labels"]
            op_target = batch["operation_labels"]
            repl_target = batch["replacement_targets"]
            valid_det = det_target != IGNORE_INDEX
            valid_op = op_target != IGNORE_INDEX
            valid_repl = repl_target != IGNORE_INDEX

            tp = int(((det_pred == 1) & (det_target == 1) & valid_det).sum().detach().cpu())
            fp = int(((det_pred == 1) & (det_target == 0) & valid_det).sum().detach().cpu())
            fn = int(((det_pred == 0) & (det_target == 1) & valid_det).sum().detach().cpu())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            op_acc = float(((op_pred == op_target) & valid_op).sum().detach().cpu()) / max(1, int(valid_op.sum().detach().cpu()))
            repl_acc = float(((repl_pred == repl_target) & valid_repl).sum().detach().cpu()) / max(1, int(valid_repl.sum().detach().cpu()))
            utt_pred = (torch.sigmoid(outputs["utterance_logits"]) >= 0.5).long()
            utt_target = batch["utterance_labels"].long()
            utt_tp = int(((utt_pred == 1) & (utt_target == 1)).sum().detach().cpu())
            utt_fp = int(((utt_pred == 1) & (utt_target == 0)).sum().detach().cpu())
            utt_fn = int(((utt_pred == 0) & (utt_target == 1)).sum().detach().cpu())
            utt_p = utt_tp / max(1, utt_tp + utt_fp)
            utt_r = utt_tp / max(1, utt_tp + utt_fn)
            utt_f1 = 2 * utt_p * utt_r / max(1e-12, utt_p + utt_r)
            return {
                "token_detection_f1": f1,
                "token_detection_precision": precision,
                "token_detection_recall": recall,
                "operation_accuracy": op_acc,
                "replacement_accuracy_on_sub": repl_acc,
                "utterance_f1": utt_f1,
            }
        """
    ),
    md(
        """
        ## 9. Training and evaluation helpers

        The sampler balances correct and mispronounced utterances to avoid the
        canonical-copy local optimum.
        """
    ),
    code(
        r"""
        def make_balanced_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
            counts = frame["is_error"].value_counts().to_dict()
            weights = frame["is_error"].map(lambda x: 1.0 / max(1, counts[bool(x)])).astype(float).values
            return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

        def make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool, balanced: bool, has_labels: bool = True) -> DataLoader:
            ds = CMEDDataset(frame, labels_lookup=labels_lookup if has_labels else None, has_labels=has_labels)
            sampler = make_balanced_sampler(frame) if balanced else None
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle and sampler is None,
                sampler=sampler,
                num_workers=cfg.num_workers,
                collate_fn=cmed_collate,
                pin_memory=torch.cuda.is_available(),
            )

        def optimizer_for_stage_a(model: CMEDV1Model) -> torch.optim.Optimizer:
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)

        def train_one_epoch(model: CMEDV1Model, loader: DataLoader, optimizer: torch.optim.Optimizer) -> dict:
            model.train()
            totals = {}
            for batch in tqdm(loader, desc="train", leave=False):
                batch = move_batch_to_device(batch, DEVICE)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_values=batch["input_values"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    canonical_ids=batch["canonical_ids"],
                    canonical_mask=batch["canonical_mask"],
                )
                loss, logs = compute_cmed_loss(outputs, batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                logs["grad_norm"] = float(grad_norm.detach().cpu())
                for k, v in logs.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
            return {k: v / max(1, len(loader)) for k, v in totals.items()}

        @torch.no_grad()
        def collect_decode_cache(model: CMEDV1Model, loader: DataLoader) -> List[dict]:
            model.eval()
            rows = []
            for batch in tqdm(loader, desc="predict", leave=False):
                batch_dev = move_batch_to_device(batch, DEVICE)
                outputs = model(
                    input_values=batch_dev["input_values"],
                    audio_attention_mask=batch_dev["audio_attention_mask"],
                    canonical_ids=batch_dev["canonical_ids"],
                    canonical_mask=batch_dev["canonical_mask"],
                )
                op_probs = F.softmax(outputs["operation_logits"], dim=-1).detach().cpu()
                repl_probs = F.softmax(outputs["replacement_logits"], dim=-1).detach().cpu()
                repl_conf, repl_ids = repl_probs.max(dim=-1)
                supervised = token_supervised_metrics(outputs, batch_dev) if "detection_labels" in batch_dev else None

                for i, tokens in enumerate(batch["canonical_tokens"]):
                    length = len(tokens)
                    row = {
                        "id": batch["ids"][i],
                        "path": batch["paths"][i],
                        "canonical": batch["canonical"][i],
                        "canonical_tokens": tokens,
                        "op_pred": op_probs[i, :length].argmax(dim=-1).tolist(),
                        "op_sub_conf": op_probs[i, :length, OP_SUB].tolist(),
                        "replacement_ids": repl_ids[i, :length].tolist(),
                        "replacement_conf": repl_conf[i, :length].tolist(),
                        "replacement_tokens": [id_to_token[int(x)] for x in repl_ids[i, :length].tolist()],
                        "supervised_metrics": supervised,
                    }
                    if "transcript" in batch:
                        row["transcript"] = batch["transcript"][i]
                    rows.append(row)
            return rows

        def decode_cache_rows(cache_rows: List[dict], sub_threshold: float, replacement_threshold: float) -> pd.DataFrame:
            rows = []
            supervised_accum = []
            for item in cache_rows:
                pred_tokens = []
                for tok, op, op_conf, repl_tok, repl_conf in zip(
                    item["canonical_tokens"],
                    item["op_pred"],
                    item["op_sub_conf"],
                    item["replacement_tokens"],
                    item["replacement_conf"],
                ):
                    if (
                        int(op) == OP_SUB
                        and float(op_conf) >= sub_threshold
                        and float(repl_conf) >= replacement_threshold
                        and repl_tok not in SPECIAL_TOKENS
                        and repl_tok != tok
                    ):
                        pred_tokens.append(repl_tok)
                    else:
                        pred_tokens.append(tok)
                row = {
                    "id": item["id"],
                    "path": item["path"],
                    "canonical": item["canonical"],
                    "predict": " ".join(pred_tokens),
                }
                if "transcript" in item:
                    row["transcript"] = item["transcript"]
                if item.get("supervised_metrics") is not None:
                    supervised_accum.append(item["supervised_metrics"])
                rows.append(row)
            pred_df = pd.DataFrame(rows)
            if supervised_accum:
                pred_df.attrs["supervised_metrics"] = {
                    k: float(np.mean([m[k] for m in supervised_accum])) for k in supervised_accum[0]
                }
            return pred_df

        @torch.no_grad()
        def predict_loader(model: CMEDV1Model, loader: DataLoader, sub_threshold: float, replacement_threshold: float) -> pd.DataFrame:
            cache_rows = collect_decode_cache(model, loader)
            return decode_cache_rows(cache_rows, sub_threshold=sub_threshold, replacement_threshold=replacement_threshold)

        def evaluate_labeled_predictions(pred_df: pd.DataFrame) -> dict:
            metrics = mdd_metrics_from_sequences(pred_df["canonical"], pred_df["transcript"], pred_df["predict"])
            metrics.update(pred_df.attrs.get("supervised_metrics", {}))
            return metrics

        def save_checkpoint(model: CMEDV1Model, path: Path, extra: Optional[dict] = None):
            payload = {
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "vocab": vocab,
                "backbone_name": BACKBONE_NAME,
                "extra": extra or {},
            }
            torch.save(payload, path)

        def load_checkpoint(path: Path, device: torch.device = DEVICE) -> CMEDV1Model:
            payload = torch.load(path, map_location=device)
            model = CMEDV1Model(vocab_size=len(payload["vocab"]["id_to_token"]), pad_id=payload["vocab"]["pad_id"], unk_id=payload["vocab"]["unk_id"])
            model.load_state_dict(payload["model_state"])
            model.to(device)
            model.eval()
            return model
        """
    ),
    md(
        """
        ## 10. Tiny balanced overfit gate

        Do not run full training if this gate fails. Passing proves that labels,
        masks, losses, and model heads can learn the edit task.
        """
    ),
    code(
        r"""
        RUN_TINY_OVERFIT = True
        RUN_FULL_TRAINING = False

        def tiny_balanced_frame(frame: pd.DataFrame, n_correct: int = 8, n_error: int = 8) -> pd.DataFrame:
            correct = frame[~frame["is_error"]].sample(n=min(n_correct, int((~frame["is_error"]).sum())), random_state=SEED)
            error = frame[frame["is_error"]].sample(n=min(n_error, int(frame["is_error"].sum())), random_state=SEED)
            tiny = pd.concat([correct, error], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
            if len(correct) < n_correct or len(error) < n_error:
                raise ValueError("Not enough correct/error samples for the required 8+8 overfit gate.")
            return tiny

        OVERFIT_PASSED = False
        TINY_CKPT = CKPT_DIR / "tiny_overfit_passed.pt"

        if RUN_TINY_OVERFIT:
            tiny_df = tiny_balanced_frame(split_train_df)
            tiny_loader = make_loader(tiny_df, batch_size=min(cfg.train_batch_size, len(tiny_df)), shuffle=True, balanced=True, has_labels=True)
            tiny_eval_loader = make_loader(tiny_df, batch_size=cfg.eval_batch_size, shuffle=False, balanced=False, has_labels=True)

            model = CMEDV1Model(vocab_size=len(id_to_token), pad_id=pad_id, unk_id=unk_id).to(DEVICE)
            model.freeze_audio_encoder()
            optimizer = optimizer_for_stage_a(model)
            tiny_log = []
            best_gate = {}

            for epoch in range(1, cfg.tiny_epochs + 1):
                train_logs = train_one_epoch(model, tiny_loader, optimizer)
                pred_df = predict_loader(
                    model,
                    tiny_eval_loader,
                    sub_threshold=cfg.default_sub_threshold,
                    replacement_threshold=cfg.default_replacement_threshold,
                )
                metrics = evaluate_labeled_predictions(pred_df)
                row = {"epoch": epoch, **train_logs, **metrics}
                tiny_log.append(row)
                best_gate = metrics
                print(
                    f"tiny epoch {epoch:02d}",
                    "loss", round(train_logs["loss"], 4),
                    "token_f1", round(metrics.get("token_detection_f1", 0.0), 4),
                    "op_acc", round(metrics.get("operation_accuracy", 0.0), 4),
                    "repl_acc", round(metrics.get("replacement_accuracy_on_sub", 0.0), 4),
                    "utt_f1", round(metrics.get("utterance_f1", 0.0), 4),
                )
                if (
                    metrics.get("token_detection_f1", 0.0) >= 0.95
                    and metrics.get("operation_accuracy", 0.0) >= 0.95
                    and metrics.get("replacement_accuracy_on_sub", 0.0) >= 0.90
                    and metrics.get("utterance_f1", 0.0) >= 0.95
                ):
                    OVERFIT_PASSED = True
                    save_checkpoint(model, TINY_CKPT, {"gate_metrics": metrics})
                    print("Tiny overfit gate PASSED.")
                    break

            pd.DataFrame(tiny_log).to_csv(EXP_DIR / "tiny_overfit_log.csv", index=False)
            if not OVERFIT_PASSED:
                print("Tiny overfit gate FAILED. Do not run full training. Debug labels/model/loss before continuing.")
        else:
            print("Tiny overfit skipped. Full training remains blocked.")
        """
    ),
    md(
        """
        ## 11. Full C-MED V1 training

        Stage A freezes wav2vec2 and trains the C-MED edit decoder for 20
        epochs. This is the only full-training stage in this Kaggle notebook.
        After Stage A, replacement_head is refined separately on gold
        substitution positions while every other module is frozen.
        """
    ),
    code(
        r"""
        BEST_CKPT = CKPT_DIR / "best_checkpoint.pt"
        train_log = []

        def checkpoint_is_better(metrics: dict, best_metrics: Optional[dict]) -> bool:
            if best_metrics is None:
                return True
            current_key = (metrics.get("correct_diagnosis", 0), metrics.get("Score", 0.0))
            best_key = (best_metrics.get("correct_diagnosis", 0), best_metrics.get("Score", 0.0))
            return current_key > best_key

        def train_replacement_head_one_epoch(model: CMEDV1Model, loader: DataLoader, optimizer: torch.optim.Optimizer) -> dict:
            model.eval()
            model.replacement_head.train()
            totals = {}
            batches = 0
            for batch in tqdm(loader, desc="replacement-head", leave=False):
                batch = move_batch_to_device(batch, DEVICE)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_values=batch["input_values"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    canonical_ids=batch["canonical_ids"],
                    canonical_mask=batch["canonical_mask"],
                )
                loss = ce_token_loss(outputs["replacement_logits"], batch["replacement_targets"], weight=None)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.replacement_head.parameters(), cfg.grad_clip)
                optimizer.step()
                sub_positions = int((batch["replacement_targets"] != IGNORE_INDEX).sum().detach().cpu())
                totals["replacement_only_loss"] = totals.get("replacement_only_loss", 0.0) + float(loss.detach().cpu())
                totals["replacement_only_grad_norm"] = totals.get("replacement_only_grad_norm", 0.0) + float(grad_norm.detach().cpu())
                totals["gold_substitution_positions"] = totals.get("gold_substitution_positions", 0.0) + float(sub_positions)
                batches += 1
            return {k: v / max(1, batches) for k, v in totals.items()}

        if RUN_FULL_TRAINING:
            if not OVERFIT_PASSED:
                raise RuntimeError("Full training is blocked because tiny overfit gate did not pass.")

            train_loader = make_loader(split_train_df, batch_size=cfg.train_batch_size, shuffle=True, balanced=True, has_labels=True)
            cal_loader = make_loader(split_cal_df, batch_size=cfg.eval_batch_size, shuffle=False, balanced=False, has_labels=True)

            model = CMEDV1Model(vocab_size=len(id_to_token), pad_id=pad_id, unk_id=unk_id).to(DEVICE)
            best_metrics = None

            model.freeze_audio_encoder()
            optimizer = optimizer_for_stage_a(model)
            for epoch in range(1, cfg.stage_a_epochs + 1):
                logs = train_one_epoch(model, train_loader, optimizer)
                cal_pred = predict_loader(
                    model,
                    cal_loader,
                    sub_threshold=cfg.default_sub_threshold,
                    replacement_threshold=cfg.default_replacement_threshold,
                )
                metrics = evaluate_labeled_predictions(cal_pred)
                row = {"stage": "A", "epoch": epoch, **logs, **metrics}
                train_log.append(row)
                pd.DataFrame(train_log).to_csv(EXP_DIR / "train_log.csv", index=False)
                print(row)
                if checkpoint_is_better(metrics, best_metrics):
                    best_metrics = metrics
                    save_checkpoint(model, BEST_CKPT, {"metrics": metrics, "stage": "A", "epoch": epoch})

            if cfg.replacement_head_epochs > 0 and BEST_CKPT.exists():
                model = load_checkpoint(BEST_CKPT, DEVICE)
                model.freeze_except_replacement_head()
                optimizer = torch.optim.AdamW(
                    [p for p in model.replacement_head.parameters() if p.requires_grad],
                    lr=cfg.replacement_head_lr,
                    weight_decay=cfg.weight_decay,
                )
                for epoch in range(1, cfg.replacement_head_epochs + 1):
                    logs = train_replacement_head_one_epoch(model, train_loader, optimizer)
                    cal_pred = predict_loader(
                        model,
                        cal_loader,
                        sub_threshold=cfg.default_sub_threshold,
                        replacement_threshold=cfg.default_replacement_threshold,
                    )
                    metrics = evaluate_labeled_predictions(cal_pred)
                    row = {"stage": "replacement_head_gold_sub", "epoch": epoch, **logs, **metrics}
                    train_log.append(row)
                    pd.DataFrame(train_log).to_csv(EXP_DIR / "train_log.csv", index=False)
                    print(row)
                    if checkpoint_is_better(metrics, best_metrics):
                        best_metrics = metrics
                        save_checkpoint(model, BEST_CKPT, {"metrics": metrics, "stage": "replacement_head_gold_sub", "epoch": epoch})
                save_checkpoint(model, CKPT_DIR / "replacement_head_refined_checkpoint.pt", {"metrics": best_metrics, "stage": "replacement_head_gold_sub"})

            print("best calibration metrics during training:", best_metrics)
        else:
            print("RUN_FULL_TRAINING is False. Set it to True only after the overfit gate passes.")
        """
    ),
    md(
        """
        ## 12. Threshold calibration

        Tune operation and replacement thresholds on multiple speaker-safe
        calibration folds. A candidate must satisfy minimum correct_diagnosis,
        not only DER. Validation is evaluated once after thresholds are selected.
        """
    ),
    code(
        r"""
        def threshold_row_is_valid(metrics: dict) -> bool:
            return (
                metrics["recall"] >= cfg.min_recall
                and metrics["true_reject"] >= cfg.min_true_reject
                and metrics["correct_diagnosis"] >= cfg.min_correct_diagnosis
                and metrics["min_fold_correct_diagnosis"] >= cfg.min_fold_correct_diagnosis
                and metrics["canonical_copy_rate"] <= cfg.max_canonical_copy_rate
                and metrics["PER"] <= cfg.max_per
                and metrics["DER"] <= cfg.max_der
            )

        def make_calibration_folds(cal_frame: pd.DataFrame) -> List[set]:
            groups = cal_frame["speaker_prefix"].astype(str).values
            unique_groups = np.unique(groups)
            if len(unique_groups) < 2:
                return [set(cal_frame["id"].astype(str))]
            n_splits = min(cfg.calibration_folds, len(unique_groups))
            splitter = GroupKFold(n_splits=n_splits)
            folds = []
            for _, fold_idx in splitter.split(cal_frame, groups=groups):
                folds.append(set(cal_frame.iloc[fold_idx]["id"].astype(str)))
            return folds

        def aggregate_fold_metrics(fold_metrics: List[dict]) -> dict:
            mean_keys = [
                "F1",
                "DER",
                "PER",
                "Score",
                "precision",
                "recall",
                "canonical_copy_rate",
                "avg_predicted_edits_per_utterance",
                "zero_edit_prediction_rate",
            ]
            sum_keys = ["true_reject", "false_reject", "false_accept", "correct_diagnosis", "wrong_diagnosis"]
            out = {k: float(np.mean([m[k] for m in fold_metrics])) for k in mean_keys}
            for k in sum_keys:
                out[k] = int(sum(int(m[k]) for m in fold_metrics))
            out["min_fold_correct_diagnosis"] = int(min(int(m["correct_diagnosis"]) for m in fold_metrics))
            out["min_fold_recall"] = float(min(float(m["recall"]) for m in fold_metrics))
            out["num_calibration_folds"] = int(len(fold_metrics))
            return out

        def calibrate_thresholds(model: CMEDV1Model, cal_frame: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
            cal_loader = make_loader(cal_frame, batch_size=cfg.eval_batch_size, shuffle=False, balanced=False, has_labels=True)
            cache_rows = collect_decode_cache(model, cal_loader)
            folds = make_calibration_folds(cal_frame)
            cache_by_id = {str(row["id"]): row for row in cache_rows}
            rows = []
            best = None
            op_thresholds = [round(x, 2) for x in np.arange(0.20, 0.91, 0.05)]
            replacement_thresholds = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)]
            for op_threshold in op_thresholds:
                for replacement_threshold in replacement_thresholds:
                    fold_metrics = []
                    for fold_ids in folds:
                        fold_cache = [cache_by_id[sample_id] for sample_id in fold_ids if sample_id in cache_by_id]
                        pred_df = decode_cache_rows(
                            fold_cache,
                            sub_threshold=op_threshold,
                            replacement_threshold=replacement_threshold,
                        )
                        fold_metrics.append(evaluate_labeled_predictions(pred_df))
                    metrics = aggregate_fold_metrics(fold_metrics)
                    row = {
                        "sub_threshold": op_threshold,
                        "replacement_threshold": replacement_threshold,
                        **metrics,
                    }
                    row["valid_candidate"] = threshold_row_is_valid(row)
                    rows.append(row)
                    if row["valid_candidate"]:
                        current_key = (row["correct_diagnosis"], row["Score"], row["min_fold_correct_diagnosis"])
                        best_key = (-1, -1.0, -1) if best is None else (best["correct_diagnosis"], best["Score"], best["min_fold_correct_diagnosis"])
                        if current_key > best_key:
                            best = row
            if best is None:
                best = max(rows, key=lambda r: (r["correct_diagnosis"], r["recall"], r["Score"]))
                best["fallback_reason"] = "No threshold met recall/true_reject/correct_diagnosis/PER/DER/copy constraints."
            return best, pd.DataFrame(rows)

        THRESHOLDS_PATH = EXP_DIR / "thresholds.json"
        CALIBRATION_GRID_PATH = EXP_DIR / "calibration_threshold_grid.csv"

        if BEST_CKPT.exists():
            model = load_checkpoint(BEST_CKPT, DEVICE)
            best_threshold_row, threshold_grid_df = calibrate_thresholds(model, split_cal_df)
            threshold_grid_df.to_csv(CALIBRATION_GRID_PATH, index=False)
            thresholds = {
                "sub_threshold": float(best_threshold_row["sub_threshold"]),
                "replacement_threshold": float(best_threshold_row["replacement_threshold"]),
            }
            THRESHOLDS_PATH.write_text(json.dumps({"thresholds": thresholds, "selected": best_threshold_row}, indent=2), encoding="utf-8")
            print("selected thresholds:", thresholds)
            print(best_threshold_row)
        else:
            thresholds = {
                "sub_threshold": cfg.default_sub_threshold,
                "replacement_threshold": cfg.default_replacement_threshold,
            }
            print("No best checkpoint found yet. Using default thresholds:", thresholds)
        """
    ),
    md(
        """
        ## 13. Validation once

        Run this after calibration. Do not tune thresholds on validation.
        """
    ),
    code(
        r"""
        if BEST_CKPT.exists():
            model = load_checkpoint(BEST_CKPT, DEVICE)
            val_loader = make_loader(split_val_df, batch_size=cfg.eval_batch_size, shuffle=False, balanced=False, has_labels=True)
            validation_pred_df = predict_loader(
                model,
                val_loader,
                sub_threshold=thresholds["sub_threshold"],
                replacement_threshold=thresholds["replacement_threshold"],
            )
            validation_metrics = evaluate_labeled_predictions(validation_pred_df)
            validation_pred_df.to_csv(EXP_DIR / "validation_predictions.csv", index=False)
            (EXP_DIR / "best_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
            print(json.dumps(validation_metrics, indent=2))

            if validation_metrics["canonical_copy_rate"] > 0.98 and validation_metrics["recall"] < 0.10:
                raise RuntimeError("Validation collapse detected: canonical_copy_rate > 0.98 and recall < 0.10.")
        else:
            print("No best checkpoint found. Train and calibrate before validation.")
        """
    ),
    md(
        """
        ## 14. Error analysis

        Save row-level diagnostics for false negatives, false positives, and
        wrong replacements.
        """
    ),
    code(
        r"""
        def classify_errors(canonical: str, transcript: str, predict: str) -> List[str]:
            c = split_phones(canonical)
            t = split_phones(transcript)
            p = split_phones(predict)
            actual = token_errors_against_canonical(c, t)
            pred = token_errors_against_canonical(c, p)
            tags = []
            for pos in set(actual) - set(pred):
                tags.append(f"false_negative@{pos}:{actual[pos]}")
            for pos in set(pred) - set(actual):
                tags.append(f"false_positive@{pos}:{pred[pos]}")
            for pos in set(actual) & set(pred):
                if actual[pos] != pred[pos]:
                    tags.append(f"wrong_replacement@{pos}:actual={actual[pos]} pred={pred[pos]}")
            return tags

        def write_error_analysis(pred_df: pd.DataFrame, prefix: str):
            rows = []
            for _, row in pred_df.iterrows():
                tags = classify_errors(row["canonical"], row["transcript"], row["predict"])
                rows.append({
                    "id": row["id"],
                    "canonical": row["canonical"],
                    "transcript": row["transcript"],
                    "predict": row["predict"],
                    "error_tags": " | ".join(tags),
                    "num_error_tags": len(tags),
                })
            out_df = pd.DataFrame(rows)
            out_csv = REPORT_DIR / f"{prefix}_error_analysis.csv"
            out_md = REPORT_DIR / f"{prefix}_error_summary.md"
            out_df.to_csv(out_csv, index=False)
            summary = out_df["error_tags"].str.split(" | ", regex=False).explode()
            summary = summary[summary.fillna("") != ""]
            top = summary.value_counts().head(30)
            lines = ["# C-MED error summary", "", "| error_tag | count |", "|---|---:|"]
            for tag, count in top.items():
                lines.append(f"| {tag} | {int(count)} |")
            out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print("wrote:", out_csv)
            print("wrote:", out_md)

        if "validation_pred_df" in globals():
            write_error_analysis(validation_pred_df, cfg.exp_id)
        else:
            print("Run validation first to create validation_pred_df.")
        """
    ),
    md(
        """
        ## 15. Public/private inference and submission

        Public test is labeled in this workspace, so it may be used for an
        inference audit only. Do not train or tune on public/private test.

        Submission is written in the standard challenge format:

        `/kaggle/working/results.csv`

        `/kaggle/working/predict.zip`

        The zip contains exactly one file, `results.csv`, and that CSV contains
        exactly one column, `predict`. A compatibility alias
        `/kaggle/working/prediction.zip` is also saved because the previous
        Kaggle notebook used that filename.
        """
    ),
    code(
        r"""
        def prepare_test_frame(meta_path: Path, audio_roots: Sequence[Path], required_cols: set) -> pd.DataFrame:
            df = read_csv_checked(meta_path, required_cols, meta_path.name)
            df["audio_path_resolved"] = df["path"].map(lambda p: resolve_audio_path(p, audio_roots))
            missing = df[df["audio_path_resolved"].isna()]
            if len(missing):
                display(missing[["id", "path"]].head(20))
                raise FileNotFoundError(f"Some test audio paths cannot be resolved for {meta_path}")
            return df

        def predict_test_frame(
            model: CMEDV1Model,
            frame: pd.DataFrame,
            sub_threshold: float,
            replacement_threshold: float,
            has_labels: bool,
        ) -> pd.DataFrame:
            loader = DataLoader(
                CMEDDataset(frame, labels_lookup=None, has_labels=False),
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=cmed_collate,
                pin_memory=torch.cuda.is_available(),
            )
            pred_df = predict_loader(
                model,
                loader,
                sub_threshold=sub_threshold,
                replacement_threshold=replacement_threshold,
            )
            if has_labels and "transcript" in frame.columns:
                pred_df["transcript"] = frame["transcript"].values
            return pred_df

        def write_submission(source_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[Path, Path]:
            results_df = pd.DataFrame({"predict": pred_df["predict"].fillna("").astype(str).values})
            if len(results_df) != len(source_df):
                raise AssertionError("results row count does not match test metadata")
            if list(results_df.columns) != ["predict"]:
                raise AssertionError(f"results.csv must contain exactly one predict column, got {list(results_df.columns)}")
            if results_df["predict"].isna().any():
                raise AssertionError("submission contains NaN predictions")
            if (results_df["predict"].astype(str).str.strip() == "").any():
                raise AssertionError("submission contains empty predictions")

            results_path = SUBMISSION_DIR / "results.csv"
            zip_path = SUBMISSION_DIR / "predict.zip"
            compatibility_zip_path = SUBMISSION_DIR / "prediction.zip"
            manifest_path = SUBMISSION_DIR / "submission_manifest.json"
            results_df.to_csv(results_path, index=False)

            for out_zip in [zip_path, compatibility_zip_path]:
                with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(results_path, arcname="results.csv")
                with zipfile.ZipFile(out_zip) as zf:
                    names = zf.namelist()
                    assert names == ["results.csv"], names
                    zipped_results = zf.read("results.csv")

            manifest = {
                "results_path": str(results_path),
                "zip_path": str(zip_path),
                "compatibility_zip_path": str(compatibility_zip_path),
                "inner_files": ["results.csv"],
                "columns": list(results_df.columns),
                "rows": int(len(results_df)),
                "line_count_in_zip_results": int(zipped_results.count(b"\n")),
                "sha256_results_in_zip": hashlib.sha256(zipped_results).hexdigest(),
                "first_prediction": str(results_df.iloc[0]["predict"]) if len(results_df) else "",
                "last_prediction": str(results_df.iloc[-1]["predict"]) if len(results_df) else "",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print("wrote:", results_path)
            print("wrote:", zip_path)
            print("wrote compatibility alias:", compatibility_zip_path)
            print("wrote:", manifest_path)
            print("columns:", list(results_df.columns))
            return results_path, zip_path

        RUN_PUBLIC_INFERENCE = False
        RUN_PRIVATE_INFERENCE = False

        if BEST_CKPT.exists() and (RUN_PUBLIC_INFERENCE or RUN_PRIVATE_INFERENCE):
            model = load_checkpoint(BEST_CKPT, DEVICE)

            if RUN_PUBLIC_INFERENCE:
                public_df = prepare_test_frame(PUBLIC_META_PATH, [PUBLIC_ROOT, DATA_ROOT, PROJECT_ROOT], REQUIRED_TRAIN_COLUMNS)
                public_pred_df = predict_test_frame(
                    model,
                    public_df,
                    thresholds["sub_threshold"],
                    thresholds["replacement_threshold"],
                    has_labels=True,
                )
                public_metrics = evaluate_labeled_predictions(public_pred_df)
                public_pred_df.to_csv(EXP_DIR / "public_test_predictions_audit.csv", index=False)
                print("public audit metrics:", json.dumps(public_metrics, indent=2))

            if RUN_PRIVATE_INFERENCE:
                private_df = prepare_test_frame(PRIVATE_META_PATH, [PRIVATE_ROOT, DATA_ROOT, PROJECT_ROOT], REQUIRED_PRIVATE_COLUMNS)
                private_pred_df = predict_test_frame(
                    model,
                    private_df,
                    thresholds["sub_threshold"],
                    thresholds["replacement_threshold"],
                    has_labels=False,
                )
                private_pred_df.to_csv(EXP_DIR / "private_test_predictions.csv", index=False)
                write_submission(private_df, private_pred_df)
        else:
            print("Inference skipped. Train a checkpoint and set RUN_PUBLIC_INFERENCE/RUN_PRIVATE_INFERENCE to True.")
        """
    ),
    md(
        """
        ## 16. Reproducibility manifest

        Save config and file locations so the run is auditable.
        """
    ),
    code(
        r"""
        manifest = {
            "pipeline": "C-MED V1 KEEP/SUB",
            "backbone": BACKBONE_NAME,
            "no_ctc_first_pipeline": True,
            "project_root": str(PROJECT_ROOT),
            "data_root": str(DATA_ROOT),
            "train_meta_path": str(TRAIN_META_PATH),
            "public_meta_path": str(PUBLIC_META_PATH),
            "private_meta_path": str(PRIVATE_META_PATH),
            "output_root": str(OUTPUT_ROOT),
            "config": asdict(cfg),
            "artifacts": {
                "processed_dir": str(PROCESSED_DIR),
                "split_dir": str(SPLIT_DIR),
                "report_dir": str(REPORT_DIR),
                "experiment_dir": str(EXP_DIR),
                "checkpoint_dir": str(CKPT_DIR),
                "submission_dir": str(SUBMISSION_DIR),
            },
        }
        (EXP_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        """
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
