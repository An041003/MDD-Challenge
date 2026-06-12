"""Reusable C-MED components for the MDD Challenge project."""

from .alignment import IGNORE_INDEX, OP_KEEP, OP_SUB, levenshtein_alignment, make_v1_label_row
from .config import BACKBONE_NAME, CMEDConfig
from .metrics import mdd_metrics_from_sequences
from .model import CMEDV1Model
from .phonemes import build_phoneme_vocab, split_phones
from .submission import write_submission

__all__ = [
    "BACKBONE_NAME",
    "CMEDConfig",
    "CMEDV1Model",
    "IGNORE_INDEX",
    "OP_KEEP",
    "OP_SUB",
    "build_phoneme_vocab",
    "levenshtein_alignment",
    "make_v1_label_row",
    "mdd_metrics_from_sequences",
    "split_phones",
    "write_submission",
]
