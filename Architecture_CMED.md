# Architecture.md - C-MED for MDD Challenge

Last updated: 2026-06-05

## 1. Summary

This document defines the new architecture for the Mispronunciation Detection & Diagnosis Challenge.

The previous CTC-first pipeline is deprecated as the main solution because it encouraged a conservative canonical-copy strategy:

```text
predict ≈ canonical
PER very low
DER very low
F1 and recall very low
```

The new architecture is:

```text
C-MED: Canonical-conditioned Mispronunciation Edit Decoder
```

Core idea:

```text
Do not generate the whole transcript as ASR.
Predict token-level edits relative to the canonical phoneme sequence.
```

Allowed pretrained backbone:

```text
nguyenvulebinh/wav2vec2-base-vietnamese-250h
```

No other pretrained speech backbone is used.

---

## 2. Challenge objective

Official score:

```text
Score = 0.5 * F1 + 0.4 * (1 - DER) + 0.1 * (1 - PER)
```

This means:

```text
F1 is the primary target.
DER is the second target.
PER is the third target.
```

The model must explicitly learn:

```text
where the pronunciation error is
what kind of error it is
what transcript phoneme should be produced
```

---

## 3. Why CTC-first is not enough

The old pipeline:

```text
audio
→ CTC transcript prediction
→ compare/postprocess with canonical
→ DER-safe patching
→ mostly canonical output
```

Failure mode:

```text
canonical_copy_rate extremely high
recall extremely low
only a few errors detected
official score slightly above canonical baseline
but not a useful MDD model
```

This happens because most samples are correct, so copying canonical is a strong local optimum.

C-MED changes the learning target from transcript generation to edit prediction.

---

## 4. Data contract

Training metadata:

```text
metadata/train_phones.csv
```

Required columns:

```text
id
path
canonical
transcript
```

`canonical` and `transcript` are whitespace-separated phoneme tokens.

Example:

```text
canonical:  aː-0 m aː-4 ɗ aː-2
transcript: aː-0 m aː-4 ɗ aː-2
```

Submission:

```text
predict.zip
└── results.csv
```

`results.csv`:

```text
predict
<phoneme sequence>
<phoneme sequence>
...
```

Rows must match the test metadata order.

---

## 5. Architecture overview

```text
audio waveform
  → Wav2Vec2 Vietnamese audio encoder
  → audio hidden states

canonical phoneme tokens
  → phoneme embedding
  → canonical phoneme encoder
  → canonical hidden states

canonical hidden states query audio hidden states
  → cross-attention alignment
  → token-level fused states

token-level fused states
  → detection head
  → operation head
  → replacement head
  → utterance head
  → optional auxiliary CTC head

edit predictions
  → reconstruct predicted transcript
  → official MDD metrics
```

Visual form:

```text
                         ┌─────────────────────────────────────────┐
Audio waveform ─────────▶│ nguyenvulebinh/wav2vec2-base-vietnamese │
                         └───────────────────┬─────────────────────┘
                                             │ audio states
                                             ▼
Canonical phones ─▶ Phoneme Embedding ─▶ Canonical Encoder
                                             │
                                             ▼
                               Cross-Attention Alignment
                                             │
                                             ▼
                                  Token-Level Fused States
                                             │
        ┌────────────────────┬───────────────┼───────────────────┐
        ▼                    ▼               ▼                   ▼
 Detection Head       Operation Head   Replacement Head   Utterance Head
 correct/error        KEEP/SUB/DEL     phoneme target     sentence error
        │                    │               │
        └────────────────────┴───────────────┘
                         │
                         ▼
              Reconstruct predicted transcript
```

---

## 6. Backbone

Only one pretrained model is used:

```text
nguyenvulebinh/wav2vec2-base-vietnamese-250h
```

Recommended setup:

```text
feature extractor: frozen
transformer encoder: frozen for the current Kaggle V1 notebook
```

Stage A only:

```text
freeze full wav2vec2
train edit heads and canonical encoder
epochs: 20
```

Output dimension depends on the backbone hidden size and is projected into `model_dim`.

---

## 7. Canonical phoneme encoder

Input:

```text
canonical_ids: [B, L]
canonical_mask: [B, L]
```

Components:

```text
phoneme embedding
position embedding
2-layer Transformer encoder or BiLSTM
linear projection to model_dim
```

Output:

```text
canonical_states: [B, L, D]
```

Canonical dropout during training:

```text
drop_prob = 0.10 to 0.15
```

Purpose:

```text
prevent the model from blindly copying canonical
force it to use audio evidence
```

---

## 8. Cross-attention alignment

Canonical tokens attend to audio frames:

```text
Q = canonical_states
K = audio_states
V = audio_states
```

Output:

```text
aligned_audio_per_token: [B, L, D]
```

Fused representation:

```text
fused = LayerNorm(canonical_states + aligned_audio_per_token)
fused = FeedForward(fused)
```

The fused vector at position `i` answers:

```text
Did the speaker pronounce canonical phoneme i correctly?
If not, what edit should be applied?
```

---

## 9. Output heads

## 9.1. Detection head

Token-level binary classifier:

```text
0 = correct
1 = mispronounced
```

Shape:

```text
[B, L, 2]
```

Used for F1 optimization.

## 9.2. Operation head

V1:

```text
KEEP
SUBSTITUTE
```

V2:

```text
KEEP
SUBSTITUTE
DELETE
```

V3:

```text
KEEP
SUBSTITUTE
DELETE
plus separate INSERT head
```

## 9.3. Replacement head

When operation is SUBSTITUTE, predict actual transcript phoneme:

```text
replacement_id ∈ phoneme_vocab
```

Loss is applied only on substitute positions.

The Kaggle V1 notebook keeps the same architecture/checkpoint state dict and
adds a replacement-head refinement phase after Stage A:

```text
freeze all existing modules
train replacement_head only
use CE only on gold substitution positions
```

## 9.4. Utterance head

Predict whether the whole utterance has at least one error.

Input pooling should be masked mean or attention pooling over fused token states.

## 9.5. Auxiliary CTC head

Optional.

Input:

```text
audio states
```

Target:

```text
transcript phoneme sequence
```

Purpose:

```text
maintain audio-to-phoneme awareness
```

It is not the main output path. If it encourages ASR collapse or hurts edit learning, set weight to 0.

---

## 10. Label generation

Levenshtein alignment between canonical and transcript:

```text
canonical tokens  = c1 c2 c3 ...
transcript tokens = t1 t2 t3 ...
```

Operations:

```text
equal      → KEEP
substitute → SUBSTITUTE + replacement target
delete     → DELETE
insert     → gap INSERT
```

V1 label simplification:

```text
equal      → KEEP
substitute → SUBSTITUTE
delete     → ignored or mapped to special substitute-null depending config
insert     → ignored and logged
```

V2 supports DELETE.

V3 supports INSERT.

Label files:

```text
data/processed/cmed_edit_labels.csv
reports/edit_label_distribution.csv
reports/ignored_insertions.csv
```

Required columns:

```text
id
canonical
transcript
canonical_ids
transcript_ids
operation_labels
detection_labels
replacement_targets
utterance_label
gap_insert_labels
gap_insert_targets
```

---

## 11. Loss design

Default V1 loss:

```text
L =
  1.5 * L_detection
+ 1.0 * L_operation
+ 1.0 * L_replacement
+ 0.5 * L_utterance
+ 0.2 * L_aux_ctc
```

Where:

```text
L_detection  = focal loss or class-balanced CE
L_operation  = weighted CE
L_replacement = CE only where operation is SUBSTITUTE
L_utterance  = BCE with pos_weight
L_aux_ctc    = optional CTC loss
```

Recommended focal loss:

```text
gamma = 2.0
alpha = inverse class frequency
```

Class weights must be calculated from training split, not validation or test.

---

## 12. Anti-collapse mechanisms

C-MED must include these safeguards:

```text
balanced utterance sampler
token-level class weights
focal loss for detection
canonical dropout
recall-constrained checkpoint selection
edit-level filtering instead of sentence-level canonical fallback
collapse diagnostics in every epoch
```

Collapse diagnostics:

```text
canonical_copy_rate
zero_edit_prediction_rate
avg_predicted_edits_per_utterance
recall
true_reject
token_error_recall
```

Warning condition:

```text
canonical_copy_rate > 0.98 and recall < 0.10
```

If this happens, do not treat score as valid progress.

---

## 13. Training protocol

## 13.1. Split

Use speaker-safe split:

```text
train: 70%
calibration: 15%
validation: 15%
```

No speaker overlap.

Calibration is used for:

```text
threshold tuning
confidence tuning
recall constraints
```

Validation is used once for final internal reporting.

## 13.2. Tiny overfit gate

Before full training, select:

```text
8 correct samples
8 mispronounced samples
```

Train only on this tiny set.

Pass criteria V1:

```text
token_detection_f1 >= 0.95
operation_accuracy >= 0.95
replacement_accuracy_on_sub >= 0.90
utterance_f1 >= 0.95
```

If not passed, debug label generation, masking, loss, replacement target, and attention path.

## 13.3. Full training

Stage A only:

```text
freeze wav2vec2
train heads
epochs: 20
head_lr: 1e-4 to 3e-4
```

Use:

```text
num_workers = 0 on Kaggle if multiprocessing causes instability
AMP = true if stable
gradient clipping
early stopping
```

---

## 14. Decoding

V1:

```python
for i, token in enumerate(canonical_tokens):
    if (
        op[i] == "SUBSTITUTE"
        and operation_conf[i] >= sub_threshold
        and replacement_conf[i] >= replacement_threshold
    ):
        output.append(replacement_token[i])
    else:
        output.append(token)
```

V2 adds deletion:

```python
if op[i] == "DELETE" and conf[i] >= delete_threshold:
    continue
```

V3 adds insertion:

```python
if gap_insert_conf[i] >= insert_threshold:
    output.append(insert_token[i])
```

Important rule:

```text
Never fallback the whole sentence to canonical.
Only reject low-confidence edits individually.
```

---

## 15. Threshold calibration

Thresholds are tuned on multiple speaker-safe calibration folds:

```text
substitution_threshold
replacement_threshold
delete_threshold
insert_threshold
utterance_threshold
```

Search grid example:

```text
0.20, 0.25, 0.30, ..., 0.90
```

Candidate constraints:

```text
recall >= MIN_RECALL
true_reject >= MIN_TRUE_REJECT
correct_diagnosis >= MIN_CORRECT_DIAGNOSIS
min_fold_correct_diagnosis >= MIN_FOLD_CORRECT_DIAGNOSIS
PER <= MAX_PER
DER <= MAX_DER
canonical_copy_rate <= MAX_CANONICAL_COPY_RATE
```

Default:

```text
MIN_RECALL = 0.20
MIN_TRUE_REJECT = 10
MIN_CORRECT_DIAGNOSIS = 10
MIN_FOLD_CORRECT_DIAGNOSIS = 1
MAX_PER = 0.10
MAX_DER = 0.35
MAX_CANONICAL_COPY_RATE = 0.95
```

Among valid candidates, prioritize correct_diagnosis, then official score.

---

## 16. Evaluation

Use official-style MDD metric.

Required outputs:

```text
F1
DER
PER
Score
precision
recall
true_reject
false_reject
false_accept
correct_diagnosis
wrong_diagnosis
min_fold_correct_diagnosis
replacement_threshold
canonical_copy_rate
```

Important DER definition:

```text
DER = wrong_diagnosis / (correct_diagnosis + wrong_diagnosis)
```

Do not compute DER over all actual errors if the official metric differs.

Sanity checks:

```text
predict = canonical  → canonical baseline
predict = transcript → score should be 1.0 on labeled split
random edit prediction → should not outperform meaningful models
```

---

## 17. Experiments

## V1: KEEP/SUB only

```text
exp_id: cmed_v1_keep_sub
backbone: nguyenvulebinh/wav2vec2-base-vietnamese-250h
operations: KEEP, SUBSTITUTE
insert/delete: disabled
```

Goal:

```text
break canonical-copy collapse
recall >= 0.20
F1 >= 0.20
PER <= 0.10
canonical_copy_rate <= 0.95
```

## V2: KEEP/SUB/DEL

```text
exp_id: cmed_v2_keep_sub_del
operations: KEEP, SUBSTITUTE, DELETE
```

Goal:

```text
cover deletion errors
improve DER and F1 without excessive PER
```

## V3: Full edit

```text
exp_id: cmed_v3_full_edit
operations: KEEP, SUBSTITUTE, DELETE
gap operation: INSERT
```

Goal:

```text
cover full alignment operation space
```

Do not implement V2/V3 before V1 passes tiny overfit and shows non-collapsed validation behavior.

---

## 18. Artifacts

For each experiment:

```text
experiments/<exp_id>/config.yaml
experiments/<exp_id>/train_log.csv
experiments/<exp_id>/calibration_predictions.csv
experiments/<exp_id>/validation_predictions.csv
experiments/<exp_id>/best_checkpoint.pt
experiments/<exp_id>/best_metrics.json
experiments/<exp_id>/thresholds.json
experiments/<exp_id>/collapse_diagnostics.csv
reports/<exp_id>_error_analysis.csv
reports/<exp_id>_error_summary.md
/kaggle/working/results.csv
/kaggle/working/predict.zip
/kaggle/working/prediction.zip
```

Prediction diagnostic file may include:

```text
id
canonical
transcript
predict
predicted_ops
predicted_replacements
token_confidences
```

Submission file must only include:

```text
predict
```

---

## 19. Expected improvement over old pipeline

Old pipeline behavior:

```text
very low PER
very low DER
near-zero recall
canonical-copy dominant
```

C-MED expected behavior:

```text
higher recall
higher F1
controlled DER/PER
explicit edit diagnosis
less canonical-copy collapse
better explanation in defense
```

The first target is not maximum score. The first target is a non-collapsed MDD model:

```text
recall >= 0.20
canonical_copy_rate <= 0.95
token-level edit predictions not empty
```

Then optimize official score.

---

## 20. Defense explanation

Core argument:

```text
The challenge is not pure ASR. Since F1 and DER account for 90% of the score,
we model MDD as canonical-conditioned edit prediction. The model receives both
audio and the canonical phoneme sequence, aligns each canonical phoneme to
acoustic evidence, and predicts whether to keep, substitute, delete, or insert
phonemes. This directly optimizes error detection and diagnosis, while an
optional auxiliary CTC loss preserves acoustic transcription ability.
```

Why only Vietnamese wav2vec2:

```text
The dataset is Vietnamese speech with Vietnamese phoneme/tone structure.
Among the allowed pretrained models, nguyenvulebinh/wav2vec2-base-vietnamese-250h
is the most language-matched backbone, so the redesigned pipeline focuses on
this backbone instead of spending compute on weaker ablations.
```

Why not canonical fallback:

```text
Whole-sentence canonical fallback artificially keeps PER and DER low but
destroys recall. C-MED filters edits individually, so confident local errors
can still be predicted while low-confidence edits are rejected.
```

---

## 21. Current notebook implementation

The clean notebook implementation is:

```text
notebooks/MDD_CMED_Kaggle.ipynb
```

It is generated reproducibly by:

```text
scripts/create_cmed_kaggle_notebook.py
```

Runtime is Kaggle-only. The expected input roots are:

```text
/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-training-set/MDD-Challenge-2025-training-set
/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-public-test/MDD-Challenge-2025-public-test
/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-private-test/MDD-Challenge-2025-private-test
```

All artifacts are written under:

```text
/kaggle/working
```

Notebook scope:

```text
C-MED V1 KEEP/SUB only
no CTC-first transcript generation
no whole-sentence canonical fallback
full training blocked until the tiny balanced overfit gate passes
DELETE and INSERT are logged but not trained in V1
```

Submission note:

```text
The notebook writes standard predict-only results.csv directly to /kaggle/working.
The primary zip is /kaggle/working/predict.zip. A compatibility alias
/kaggle/working/prediction.zip is also saved for workflows that expect the
older filename.
```
