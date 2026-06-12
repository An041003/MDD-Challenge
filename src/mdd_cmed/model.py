import torch
import torch.nn as nn
from transformers import AutoModel

from .config import BACKBONE_NAME, CMEDConfig


class CMEDV1Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        unk_id: int,
        cfg: CMEDConfig | None = None,
        backbone_name: str = BACKBONE_NAME,
    ):
        super().__init__()
        self.cfg = cfg or CMEDConfig()
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.audio_encoder = AutoModel.from_pretrained(backbone_name)
        audio_dim = self.audio_encoder.config.hidden_size

        self.audio_proj = nn.Linear(audio_dim, self.cfg.model_dim)
        self.phone_embedding = nn.Embedding(vocab_size, self.cfg.model_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(self.cfg.max_canonical_len, self.cfg.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.model_dim,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.model_dim * 4,
            dropout=self.cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.canonical_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.canon_layers)
        self.cross_attention = nn.MultiheadAttention(
            self.cfg.model_dim,
            self.cfg.num_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.fuse_norm = nn.LayerNorm(self.cfg.model_dim)
        self.fuse_ffn = nn.Sequential(
            nn.Linear(self.cfg.model_dim, self.cfg.model_dim * 4),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.model_dim * 4, self.cfg.model_dim),
            nn.Dropout(self.cfg.dropout),
        )
        self.out_norm = nn.LayerNorm(self.cfg.model_dim)

        self.detection_head = nn.Linear(self.cfg.model_dim, 2)
        self.operation_head = nn.Linear(self.cfg.model_dim, 2)
        self.replacement_head = nn.Linear(self.cfg.model_dim, vocab_size)
        self.utterance_head = nn.Sequential(
            nn.Linear(self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.model_dim, 1),
        )

    def freeze_audio_encoder(self) -> None:
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

    def freeze_except_replacement_head(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for param in self.replacement_head.parameters():
            param.requires_grad = True

    def _audio_feature_mask(self, feature_len: int, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if hasattr(self.audio_encoder, "_get_feature_vector_attention_mask"):
            return self.audio_encoder._get_feature_vector_attention_mask(feature_len, attention_mask)
        return None

    def _apply_canonical_dropout(self, canonical_ids: torch.Tensor, canonical_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cfg.canonical_dropout <= 0:
            return canonical_ids
        drop = torch.rand_like(canonical_ids.float()) < self.cfg.canonical_dropout
        drop = drop & canonical_mask & (canonical_ids != self.pad_id)
        out = canonical_ids.clone()
        out[drop] = self.unk_id
        return out

    def forward(
        self,
        input_values: torch.Tensor,
        audio_attention_mask: torch.Tensor | None,
        canonical_ids: torch.Tensor,
        canonical_mask: torch.Tensor,
    ) -> dict:
        audio_out = self.audio_encoder(input_values=input_values, attention_mask=audio_attention_mask)
        audio_states = self.audio_proj(audio_out.last_hidden_state)
        audio_feature_mask = self._audio_feature_mask(audio_states.shape[1], audio_attention_mask)
        audio_key_padding_mask = None if audio_feature_mask is None else ~audio_feature_mask.bool()

        canonical_ids = self._apply_canonical_dropout(canonical_ids, canonical_mask)
        bsz, seq_len = canonical_ids.shape
        if seq_len > self.cfg.max_canonical_len:
            raise ValueError(f"canonical length {seq_len} exceeds max_canonical_len={self.cfg.max_canonical_len}")
        positions = torch.arange(seq_len, device=canonical_ids.device).unsqueeze(0).expand(bsz, seq_len)
        canon_states = self.phone_embedding(canonical_ids) + self.position_embedding(positions)
        canon_states = self.canonical_encoder(canon_states, src_key_padding_mask=~canonical_mask.bool())

        attended_audio, _ = self.cross_attention(
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
