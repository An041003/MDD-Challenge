from dataclasses import dataclass


BACKBONE_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"


@dataclass
class CMEDConfig:
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
